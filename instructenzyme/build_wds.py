#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
import webdataset as wds
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export InstructEnzyme index records into WebDataset shards.")
    parser.add_argument("--index_dir", type=Path, default=Path("/home/ubuntu/cqr_files/protein_design/instructenzyme/data/index"))
    parser.add_argument("--output_dir", type=Path, default=Path("/home/ubuntu/cqr_files/protein_design/instructenzyme/data/wds"))
    parser.add_argument("--maxcount", type=int, default=1000)
    parser.add_argument("--splits", type=str, default="train,val,test")
    return parser.parse_args()


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tensor_to_bytes(payload: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    summary = {}
    for split in splits:
        index_path = args.index_dir / f"{split}.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(split_dir / f"{split}-%06d.tar")
        count = 0
        with wds.ShardWriter(pattern, maxcount=args.maxcount) as sink:
            for record in tqdm(load_jsonl(index_path), desc=f"exporting {split}"):
                emb_payload = torch.load(record["embedding_path"], map_location="cpu")
                sample = {
                    "__key__": record["id"],
                    "json": json.dumps(record).encode("utf-8"),
                    "pth": tensor_to_bytes(emb_payload),
                }
                sink.write(sample)
                count += 1
        summary[split] = {"count": count, "pattern": pattern}

    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
