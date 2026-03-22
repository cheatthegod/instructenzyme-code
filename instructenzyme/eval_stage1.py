#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from instructenzyme.dataset import ProteinDataCollator, ProteinIndexDataset
from instructenzyme.modeling import InstructEnzymeStage1Model
from instructenzyme.train_stage1 import evaluate, get_amino_acid_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Stage-1 InstructEnzyme projector checkpoint.")
    parser.add_argument("--model_name_or_path", type=str, default="/home/ubuntu/cqr_files/protein_design/progen2-base")
    parser.add_argument("--projector_ckpt", type=Path, required=True)
    parser.add_argument("--index_path", type=Path, default=Path("/home/ubuntu/cqr_files/protein_design/instructenzyme/data/index/val.jsonl"))
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    amino_acid_token_ids = get_amino_acid_token_ids(tokenizer)

    ckpt = torch.load(args.projector_ckpt, map_location="cpu")
    structure_hidden_size = int(ckpt.get("structure_hidden_size", 128))
    num_queries = int(ckpt.get("num_queries", 256))
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    model = InstructEnzymeStage1Model(
        model_name_or_path=args.model_name_or_path,
        structure_hidden_size=structure_hidden_size,
        num_queries=num_queries,
        dtype=dtype,
    )
    model.projector.load_state_dict(ckpt["projector"], strict=True)
    model.to(device)

    dataset = ProteinIndexDataset(args.index_path, tokenizer, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=ProteinDataCollator(tokenizer),
        drop_last=False,
    )

    metrics = evaluate(model, dataloader, device, args.bf16, amino_acid_token_ids)
    metrics.update(
        {
            "model_name_or_path": args.model_name_or_path,
            "projector_ckpt": str(args.projector_ckpt.resolve()),
            "index_path": str(args.index_path.resolve()),
            "num_samples": len(dataset),
            "batch_size": args.batch_size,
            "device": str(device),
        }
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
