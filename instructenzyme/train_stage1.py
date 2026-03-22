#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import timedelta
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from instructenzyme.dataset import IGNORE_INDEX, ProteinDataCollator, ProteinIndexDataset
from instructenzyme.modeling import InstructEnzymeStage1Model

VALID_AAS = "ACDEFGHIKLMNPQRSTVWY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 adapter-only training for InstructEnzyme.")
    parser.add_argument("--model_name_or_path", type=str, default="/home/ubuntu/cqr_files/protein_design/progen2-base")
    parser.add_argument("--train_index", type=Path, default=Path("/home/ubuntu/cqr_files/protein_design/instructenzyme/data/index/train.jsonl"))
    parser.add_argument("--val_index", type=Path, default=Path("/home/ubuntu/cqr_files/protein_design/instructenzyme/data/index/val.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("/home/ubuntu/cqr_files/protein_design/instructenzyme/runs/progen2-base-stage1"))
    parser.add_argument("--projector_init_ckpt", type=Path, default=None)
    parser.add_argument("--structure_hidden_size", type=int, default=128)
    parser.add_argument("--num_queries", type=int, default=256)
    parser.add_argument("--projector_num_heads", type=int, default=8)
    parser.add_argument("--projector_num_layers", type=int, default=1)
    parser.add_argument("--projector_ffn_mult", type=float, default=4.0)
    parser.add_argument("--projector_dropout", type=float, default=0.0)
    parser.add_argument("--projector_pos_encoding", type=str, default="1d")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def rank0_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


def reduce_scalar(value: torch.Tensor) -> float:
    if not dist.is_initialized():
        return float(value.item())
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return float(reduced.item())


def reduce_sum_tensor(value: torch.Tensor) -> torch.Tensor:
    reduced = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def get_amino_acid_token_ids(tokenizer) -> torch.Tensor:
    token_ids = []
    for aa in VALID_AAS:
        ids = tokenizer(aa, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"expected single token for amino acid {aa}, got {ids}")
        token_ids.append(ids[0])
    return torch.tensor(sorted(set(token_ids)), dtype=torch.long)


@torch.no_grad()
def evaluate(model, dataloader, device, use_bf16: bool, amino_acid_token_ids: torch.Tensor) -> Dict[str, float]:
    model.eval()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    aa_token_ids = amino_acid_token_ids.to(device)

    total_loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    total_valid_tokens = torch.zeros(1, device=device, dtype=torch.float64)
    total_recovery_tokens = torch.zeros(1, device=device, dtype=torch.float64)
    total_top1_correct = torch.zeros(1, device=device, dtype=torch.float64)
    total_top5_correct = torch.zeros(1, device=device, dtype=torch.float64)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                structure_embeddings=batch["structure_embeddings"],
                structure_attention_mask=batch["structure_attention_mask"],
            )

        logits = outputs.logits.detach().float()
        prompt_len = logits.shape[1] - batch["labels"].shape[1]
        prompt_labels = torch.full(
            (batch["labels"].shape[0], prompt_len),
            IGNORE_INDEX,
            dtype=batch["labels"].dtype,
            device=batch["labels"].device,
        )
        full_labels = torch.cat([prompt_labels, batch["labels"]], dim=1)

        shift_logits = logits[:, :-1, :]
        shift_labels = full_labels[:, 1:]
        valid_mask = shift_labels.ne(IGNORE_INDEX)
        valid_count = valid_mask.sum()

        if valid_count.item() == 0:
            continue

        total_loss_sum += outputs.loss.detach().to(torch.float64) * valid_count.to(torch.float64)
        total_valid_tokens += valid_count.to(torch.float64)

        pred_ids = shift_logits.argmax(dim=-1)
        aa_mask = valid_mask & torch.isin(shift_labels, aa_token_ids)
        aa_count = aa_mask.sum()
        total_recovery_tokens += aa_count.to(torch.float64)

        if aa_count.item() > 0:
            top1_correct = (pred_ids.eq(shift_labels) & aa_mask).sum()
            topk = shift_logits.topk(k=min(5, shift_logits.shape[-1]), dim=-1).indices
            top5_correct = topk.eq(shift_labels.unsqueeze(-1)).any(dim=-1)
            top5_correct = (top5_correct & aa_mask).sum()
            total_top1_correct += top1_correct.to(torch.float64)
            total_top5_correct += top5_correct.to(torch.float64)

    total_loss_sum = reduce_sum_tensor(total_loss_sum)
    total_valid_tokens = reduce_sum_tensor(total_valid_tokens)
    total_recovery_tokens = reduce_sum_tensor(total_recovery_tokens)
    total_top1_correct = reduce_sum_tensor(total_top1_correct)
    total_top5_correct = reduce_sum_tensor(total_top5_correct)

    if total_valid_tokens.item() == 0:
        return {
            "val_loss": float("nan"),
            "val_ppl": float("nan"),
            "val_recovery": float("nan"),
            "val_top5_recovery": float("nan"),
            "val_token_count": 0.0,
            "val_recovery_token_count": 0.0,
        }

    val_loss = float((total_loss_sum / total_valid_tokens).item())
    val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
    if total_recovery_tokens.item() > 0:
        val_recovery = float((total_top1_correct / total_recovery_tokens).item())
        val_top5_recovery = float((total_top5_correct / total_recovery_tokens).item())
    else:
        val_recovery = float("nan")
        val_top5_recovery = float("nan")

    return {
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "val_recovery": val_recovery,
        "val_top5_recovery": val_top5_recovery,
        "val_token_count": float(total_valid_tokens.item()),
        "val_recovery_token_count": float(total_recovery_tokens.item()),
    }


def save_checkpoint(model, output_dir: Path, tag: str, metrics: Dict):
    module = model.module if isinstance(model, DDP) else model
    ckpt_dir = output_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    module.save_projector(ckpt_dir, extra_state={"metrics": metrics})
    with (ckpt_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()
    set_seed(args.seed + rank)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    amino_acid_token_ids = get_amino_acid_token_ids(tokenizer)

    train_dataset = ProteinIndexDataset(args.train_index, tokenizer, max_samples=args.max_train_samples)
    val_dataset = ProteinIndexDataset(args.val_index, tokenizer, max_samples=args.max_val_samples)
    collator = ProteinDataCollator(tokenizer)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = InstructEnzymeStage1Model(
        model_name_or_path=args.model_name_or_path,
        structure_hidden_size=args.structure_hidden_size,
        num_queries=args.num_queries,
        num_heads=args.projector_num_heads,
        num_layers=args.projector_num_layers,
        ffn_mult=args.projector_ffn_mult,
        dropout=args.projector_dropout,
        pos_encoding=args.projector_pos_encoding,
        dtype=dtype,
    )
    if args.projector_init_ckpt is not None:
        ckpt_path = args.projector_init_ckpt
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / 'projector.pt'
        init_state = torch.load(ckpt_path, map_location='cpu')
        projector_state = init_state['projector'] if isinstance(init_state, dict) and 'projector' in init_state else init_state
        model.projector.load_state_dict(projector_state, strict=True)
        rank0_print(f"loaded projector init from {ckpt_path}")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    trainable_params = model.module.get_trainable_parameters() if isinstance(model, DDP) else model.get_trainable_parameters()
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    total_train_steps = args.max_train_steps if args.max_train_steps > 0 else steps_per_epoch * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_train_steps,
    )

    rank0_print(f"train_samples={len(train_dataset)} val_samples={len(val_dataset)} world_size={world_size}")
    rank0_print(f"steps_per_epoch={steps_per_epoch} total_train_steps={total_train_steps}")

    log_path = args.output_dir / "train_log.jsonl"
    best_val = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_iterator = tqdm(train_loader, disable=rank != 0, desc=f"epoch {epoch}")
        for batch_idx, batch in enumerate(epoch_iterator):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    structure_embeddings=batch["structure_embeddings"],
                    structure_attention_mask=batch["structure_attention_mask"],
                )
                loss = outputs.loss / args.gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                reduced_loss = reduce_scalar(loss.detach() * args.gradient_accumulation_steps)
                if rank == 0 and global_step % args.log_every == 0:
                    record = {
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": reduced_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "time": time.time(),
                    }
                    with log_path.open("a") as f:
                        f.write(json.dumps(record) + "\n")
                    epoch_iterator.set_postfix(loss=f"{reduced_loss:.4f}")

                if global_step % args.eval_every == 0 or global_step == total_train_steps:
                    metrics = evaluate(model, val_loader, device, args.bf16, amino_acid_token_ids)
                    metrics.update(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                    rank0_print(json.dumps(metrics))
                    if rank == 0:
                        with log_path.open("a") as f:
                            f.write(json.dumps(metrics) + "\n")
                        save_checkpoint(model, args.output_dir, f"step-{global_step}", metrics)
                        if metrics["val_loss"] < best_val:
                            best_val = metrics["val_loss"]
                            save_checkpoint(model, args.output_dir, "best", metrics)

                if global_step % args.save_every == 0 and rank == 0:
                    save_checkpoint(model, args.output_dir, "latest", {"step": global_step, "epoch": epoch})

                if global_step >= total_train_steps:
                    break

        if global_step >= total_train_steps:
            break

    if rank == 0:
        save_checkpoint(model, args.output_dir, "final", {"step": global_step, "best_val": best_val})
    cleanup_distributed()


if __name__ == "__main__":
    main()
