#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Tuple

import torch
from tqdm import tqdm

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
RES3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sequence/embedding manifest for InstructEnzyme training.")
    parser.add_argument("--pdb_dir", type=Path, default=Path("/home/ubuntu/cqr_files/protein_design/enzyme_pdb"))
    parser.add_argument("--embedding_dir", type=Path, default=Path("/home/ubuntu/cqr_files/protein_design/ligandmpnn_emb"))
    parser.add_argument("--output_dir", type=Path, default=Path("/home/ubuntu/cqr_files/protein_design/instructenzyme/data/index"))
    parser.add_argument("--train_frac", type=float, default=0.98)
    parser.add_argument("--val_frac", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def stable_split(sample_id: str, train_frac: float, val_frac: float) -> str:
    bucket = int(hashlib.md5(sample_id.encode("utf-8")).hexdigest(), 16) % 10000
    train_cut = int(train_frac * 10000)
    val_cut = int((train_frac + val_frac) * 10000)
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def extract_single_chain_sequence(pdb_path: Path) -> Tuple[str, str]:
    chains: dict[str, list[str]] = {}
    seen_residues: dict[str, set[tuple[str, str]]] = {}
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A", "1"}:
                continue
            resname = line[17:20].strip().upper()
            if resname not in RES3_TO_1:
                continue
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip() or "_"
            key = (resseq, icode)
            chain_seen = seen_residues.setdefault(chain_id, set())
            if key in chain_seen:
                continue
            chain_seen.add(key)
            chains.setdefault(chain_id, []).append(RES3_TO_1[resname])

    if len(chains) != 1:
        raise ValueError(f"expected exactly one protein chain, found {len(chains)}")
    chain_id, residues = next(iter(chains.items()))
    seq = "".join(residues)
    if not seq:
        raise ValueError("empty protein sequence")
    if any(aa not in VALID_AAS for aa in seq):
        raise ValueError("sequence contains non-canonical residues")
    return chain_id, seq


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(args.pdb_dir.glob("*.pdb"))
    if args.limit > 0:
        pdb_files = pdb_files[: args.limit]

    split_records = {"train": [], "val": [], "test": []}
    stats = Counter()

    for pdb_path in tqdm(pdb_files, desc="building manifest"):
        sample_id = pdb_path.stem
        stats["pdb_total"] += 1
        emb_path = args.embedding_dir / f"{sample_id}.pt"
        if not emb_path.exists():
            stats["missing_embedding"] += 1
            continue

        try:
            chain_id, seq = extract_single_chain_sequence(pdb_path)
        except Exception as exc:
            stats[f"pdb_error::{type(exc).__name__}"] += 1
            continue

        try:
            emb_payload = torch.load(emb_path, map_location="cpu")
            h_v = emb_payload["h_V_last_layer"]
        except Exception as exc:
            stats[f"embedding_error::{type(exc).__name__}"] += 1
            continue

        if h_v.ndim != 2:
            stats["bad_embedding_rank"] += 1
            continue
        if h_v.shape[0] != len(seq):
            stats["length_mismatch"] += 1
            continue

        split = stable_split(sample_id, args.train_frac, args.val_frac)
        record = {
            "id": sample_id,
            "split": split,
            "chain_id": chain_id,
            "sequence": seq,
            "seq_len": len(seq),
            "embedding_dim": int(h_v.shape[1]),
            "pdb_path": str(pdb_path.resolve()),
            "embedding_path": str(emb_path.resolve()),
        }
        split_records[split].append(record)
        stats["usable"] += 1
        stats[f"usable::{split}"] += 1

    for split, records in split_records.items():
        with (args.output_dir / f"{split}.jsonl").open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    with (args.output_dir / "all.jsonl").open("w") as f:
        for split in ("train", "val", "test"):
            for record in split_records[split]:
                f.write(json.dumps(record) + "\n")

    stats_payload = {
        "pdb_dir": str(args.pdb_dir.resolve()),
        "embedding_dir": str(args.embedding_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "stats": dict(stats),
        "split_sizes": {split: len(records) for split, records in split_records.items()},
    }
    with (args.output_dir / "stats.json").open("w") as f:
        json.dump(stats_payload, f, indent=2)

    print(json.dumps(stats_payload, indent=2))


if __name__ == "__main__":
    main()
