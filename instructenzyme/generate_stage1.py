#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from instructenzyme.dataset import ProteinIndexDataset
from instructenzyme.modeling import InstructEnzymeStage1Model
from instructenzyme.train_stage1 import get_amino_acid_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate protein sequences from a Stage-1 projector-conditioned model.")
    parser.add_argument("--model_name_or_path", type=str, default="/home/ubuntu/cqr_files/protein_design/progen2-base")
    parser.add_argument("--projector_ckpt", type=Path, required=True)
    parser.add_argument("--index_path", type=Path, default=Path("/home/ubuntu/cqr_files/protein_design/instructenzyme/data/index/test.jsonl"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=0, help="0 means max(native_length in batch) + 1")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chunked(items: list[int], batch_size: int) -> list[list[int]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def decode_sequence(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False).replace(" ", "")


def restrict_logits(logits: torch.Tensor, allowed_ids: torch.Tensor) -> torch.Tensor:
    filtered = torch.full_like(logits, -1e9)
    filtered[:, allowed_ids] = logits[:, allowed_ids]
    return filtered


def nucleus_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    sorted_mask = cumulative > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_mask, -1e9)
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    next_ids = sorted_indices.gather(-1, sampled_idx)
    return next_ids


def collate_structure_batch(samples: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(sample["structure_embeddings"].shape[0] for sample in samples)
    hidden_dim = samples[0]["structure_embeddings"].shape[1]
    batch = torch.zeros((len(samples), max_len, hidden_dim), dtype=torch.float32, device=device)
    mask = torch.zeros((len(samples), max_len), dtype=torch.bool, device=device)
    for i, sample in enumerate(samples):
        emb = sample["structure_embeddings"].to(device)
        length = emb.shape[0]
        batch[i, :length] = emb
        mask[i, :length] = True
    return batch, mask


@torch.no_grad()
def generate_batch(
    model: InstructEnzymeStage1Model,
    tokenizer,
    structure_embeddings: torch.Tensor,
    structure_attention_mask: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    allowed_ids: torch.Tensor,
    per_sample_max_new_tokens: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
    use_bf16: bool,
) -> tuple[list[str], list[bool]]:
    device = structure_embeddings.device
    batch_size = structure_embeddings.shape[0]
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    per_sample_max_new_tokens = per_sample_max_new_tokens.to(device=device, dtype=torch.long)
    max_decode_steps = int(per_sample_max_new_tokens.max().item()) if per_sample_max_new_tokens.numel() > 0 else 0

    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
        prompt_embeds = model.encode_structure(structure_embeddings, structure_attention_mask)
        start_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        start_embeds = model.get_input_embeddings()(start_ids)
        inputs_embeds = torch.cat([prompt_embeds, start_embeds], dim=1)
        attention_mask = torch.ones((batch_size, inputs_embeds.shape[1]), dtype=torch.long, device=device)
        outputs = model.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :].float()
    generated_ids: list[list[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    ended_with_stop = [False for _ in range(batch_size)]

    for step_idx in range(max_decode_steps):
        filtered_logits = restrict_logits(logits, allowed_ids)
        if do_sample:
            if temperature <= 0:
                raise ValueError("temperature must be > 0 when sampling")
            filtered_logits = filtered_logits / temperature
            if top_p < 1.0:
                next_tokens = nucleus_sample(filtered_logits, top_p)
            else:
                probs = torch.softmax(filtered_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = filtered_logits.argmax(dim=-1, keepdim=True)

        for i in range(batch_size):
            if finished[i]:
                continue
            token_id = int(next_tokens[i, 0].item())
            if token_id == end_token_id:
                finished[i] = True
                ended_with_stop[i] = True
                continue

            generated_ids[i].append(token_id)
            if (step_idx + 1) >= int(per_sample_max_new_tokens[i].item()):
                finished[i] = True

        if bool(finished.all().item()):
            break

        feed_tokens = next_tokens.clone()
        feed_tokens[finished] = end_token_id
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
            outputs = model.backbone(
                input_ids=feed_tokens,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()

    generated_sequences = [decode_sequence(tokenizer, ids) for ids in generated_ids]
    return generated_sequences, ended_with_stop


def main() -> None:
    args = parse_args()
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    set_seed(args.seed + args.shard_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    start_token_id = tokenizer("1", add_special_tokens=False)["input_ids"][0]
    end_token_id = tokenizer("2", add_special_tokens=False)["input_ids"][0]
    amino_acid_token_ids = get_amino_acid_token_ids(tokenizer)
    allowed_ids = torch.tensor(sorted(set(amino_acid_token_ids.tolist() + [end_token_id])), dtype=torch.long, device=device)

    ckpt = torch.load(args.projector_ckpt, map_location="cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = InstructEnzymeStage1Model(
        model_name_or_path=args.model_name_or_path,
        structure_hidden_size=int(ckpt.get("structure_hidden_size", 128)),
        num_queries=int(ckpt.get("num_queries", 256)),
        dtype=dtype,
    )
    model.projector.load_state_dict(ckpt["projector"], strict=True)
    model.backbone.config.use_cache = True
    model.to(device)
    model.eval()

    dataset = ProteinIndexDataset(args.index_path, tokenizer, max_samples=args.max_samples)
    shard_indices = [i for i in range(len(dataset)) if i % args.num_shards == args.shard_index]
    shard_indices.sort(key=lambda i: int(dataset.records[i].get("seq_len", len(dataset.records[i]["sequence"]))))
    shard_batches = chunked(shard_indices, args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / f"shard-{args.shard_index:02d}.jsonl"
    summary_path = args.output_dir / f"shard-{args.shard_index:02d}-summary.json"

    total_native_len = 0
    total_generated_len = 0
    total_matches = 0
    exact_match_count = 0
    stop_count = 0
    per_seq_recovery_sum = 0.0
    length_ratio_sum = 0.0
    count = 0

    with records_path.open("w") as records_file:
        pbar = tqdm(total=len(shard_indices), desc=f"generate shard {args.shard_index}/{args.num_shards}")
        for batch_indices in shard_batches:
            samples = [dataset[idx] for idx in batch_indices]
            native_sequences = [sample["sequence"] for sample in samples]
            native_lengths = [len(seq) for seq in native_sequences]
            if args.max_new_tokens > 0:
                per_sample_max_new_tokens = torch.full((len(samples),), args.max_new_tokens, dtype=torch.long, device=device)
            else:
                per_sample_max_new_tokens = torch.tensor([length + 1 for length in native_lengths], dtype=torch.long, device=device)
            structure_embeddings, structure_attention_mask = collate_structure_batch(samples, device)

            generated_sequences, ended_with_stop = generate_batch(
                model=model,
                tokenizer=tokenizer,
                structure_embeddings=structure_embeddings,
                structure_attention_mask=structure_attention_mask,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                allowed_ids=allowed_ids,
                per_sample_max_new_tokens=per_sample_max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                use_bf16=args.bf16,
            )

            for sample, native_sequence, native_length, generated_sequence, did_stop in zip(
                samples, native_sequences, native_lengths, generated_sequences, ended_with_stop
            ):
                generated_length = len(generated_sequence)
                matches = sum(int(a == b) for a, b in zip(native_sequence, generated_sequence))
                recovery = matches / native_length if native_length > 0 else float("nan")
                length_ratio = generated_length / native_length if native_length > 0 else float("nan")
                exact_match = generated_sequence == native_sequence

                record = {
                    "id": sample["sample_id"],
                    "native_sequence": native_sequence,
                    "generated_sequence": generated_sequence,
                    "native_length": native_length,
                    "generated_length": generated_length,
                    "matches": matches,
                    "recovery": recovery,
                    "length_ratio": length_ratio,
                    "exact_match": exact_match,
                    "ended_with_stop": did_stop,
                    "mode": "sample" if args.do_sample else "greedy",
                    "batch_size": args.batch_size,
                }
                records_file.write(json.dumps(record) + "\n")

                total_native_len += native_length
                total_generated_len += generated_length
                total_matches += matches
                exact_match_count += int(exact_match)
                stop_count += int(did_stop)
                per_seq_recovery_sum += recovery
                length_ratio_sum += length_ratio
                count += 1

            records_file.flush()
            pbar.update(len(batch_indices))
        pbar.close()

    summary = {
        "count": count,
        "mode": "sample" if args.do_sample else "greedy",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "batch_size": args.batch_size,
        "model_name_or_path": args.model_name_or_path,
        "projector_ckpt": str(args.projector_ckpt.resolve()),
        "index_path": str(args.index_path.resolve()),
        "records_path": str(records_path.resolve()),
        "mean_sequence_recovery": per_seq_recovery_sum / count if count else float("nan"),
        "global_residue_recovery": total_matches / total_native_len if total_native_len else float("nan"),
        "exact_match_rate": exact_match_count / count if count else float("nan"),
        "stop_rate": stop_count / count if count else float("nan"),
        "mean_native_length": total_native_len / count if count else float("nan"),
        "mean_generated_length": total_generated_len / count if count else float("nan"),
        "mean_length_ratio": length_ratio_sum / count if count else float("nan"),
        "total_native_length": total_native_len,
        "total_generated_length": total_generated_len,
        "total_matches": total_matches,
        "device": str(device),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
