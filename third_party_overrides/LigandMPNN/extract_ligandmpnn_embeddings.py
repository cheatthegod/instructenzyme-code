#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from data_utils import featurize, parse_PDB
from model_utils import ProteinMPNN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract ligand-aware residue embeddings from PDB complexes using LigandMPNN."
    )
    parser.add_argument(
        "--pdb_dir",
        type=Path,
        default=Path("/home/ubuntu/cqr_files/protein_design/enzyme_pdb"),
        help="Directory containing input .pdb files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/ubuntu/cqr_files/protein_design/ligandmpnn_emb"),
        help="Directory to write output .pt embedding files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/ubuntu/cqr_files/protein_design/LigandMPNN/model_params/ligandmpnn_v_32_005_25.pt"),
        help="LigandMPNN checkpoint path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N PDB files before sharding. Use 0 or a negative value for all files.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards/processes to split the PDB list into.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Shard index for this process, in [0, num_shards).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of complexes to pad and encode together on one GPU.",
    )
    parser.add_argument(
        "--sort_by_length",
        action="store_true",
        help="Sort shard-local files by residue length estimate to reduce padding waste.",
    )
    parser.add_argument(
        "--cutoff_for_score",
        type=float,
        default=8.0,
        help="Residue-to-context cutoff used in featurize().",
    )
    parser.add_argument(
        "--use_atom_context",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to use ligand atom context.",
    )
    parser.add_argument(
        "--ligand_mpnn_use_side_chain_context",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to include side-chain atoms as additional context.",
    )
    parser.add_argument(
        "--parse_all_atoms",
        action="store_true",
        help="Parse all protein atoms when reading the PDB.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=37,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use: "auto", "cpu", or "cuda".',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files.",
    )
    parser.add_argument(
        "--save_with_metadata",
        action="store_true",
        help="If set, keep extra metadata fields instead of saving only h_V_last_layer.",
    )
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[ProteinMPNN, int]:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    atom_context_num = checkpoint["atom_context_num"]
    k_neighbors = checkpoint["num_edges"]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=args.ligand_mpnn_use_side_chain_context,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, atom_context_num


def shard_files(files: list[Path], num_shards: int, shard_index: int) -> list[Path]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return [p for idx, p in enumerate(files) if idx % num_shards == shard_index]


def estimate_length_from_pdb(pdb_path: Path) -> int:
    count = 0
    seen = set()
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip() or "_"
            key = (chain, resseq, icode)
            if key not in seen:
                seen.add(key)
                count += 1
    return count


def pad_and_stack(tensors: list[torch.Tensor], pad_value=0) -> torch.Tensor:
    max_len = max(t.shape[0] for t in tensors)
    out = []
    for t in tensors:
        if t.shape[0] == max_len:
            out.append(t)
            continue
        pad_shape = (max_len - t.shape[0],) + tuple(t.shape[1:])
        pad = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
        out.append(torch.cat([t, pad], dim=0))
    return torch.stack(out, dim=0)


def collate_feature_dicts(feature_dicts: list[dict], device: torch.device) -> tuple[dict, list[str]]:
    names = [fd["name"] for fd in feature_dicts]
    batch = {}

    keys = [
        "mask_XY",
        "Y",
        "Y_t",
        "Y_m",
        "R_idx",
        "R_idx_original",
        "chain_labels",
        "S",
        "chain_mask",
        "mask",
        "X",
        "xyz_37",
        "xyz_37_m",
        "side_chain_mask",
    ]
    for key in keys:
        present = [fd[key][0] for fd in feature_dicts if key in fd]
        if not present:
            continue
        pad_value = 0.0 if present[0].dtype.is_floating_point else 0
        batch[key] = pad_and_stack(present, pad_value=pad_value).to(device)

    return batch, names


def save_minimal_embeddings(h_v: torch.Tensor, batch_feature_dict: dict, output_dir: Path, names: list[str]) -> None:
    mask = batch_feature_dict.get("mask")
    for idx, name in enumerate(names):
        sample_embedding = h_v[idx].detach().cpu()
        if mask is not None:
            valid_mask = mask[idx].detach().bool().cpu()
            if valid_mask.any():
                sample_embedding = sample_embedding[valid_mask]
        torch.save({"h_V_last_layer": sample_embedding}, output_dir / f"{name}.pt")


def process_batch(model: ProteinMPNN, batch_items: list[dict], output_dir: Path, device: torch.device, save_with_metadata: bool) -> None:
    if not batch_items:
        return
    batch_feature_dict, names = collate_feature_dicts(batch_items, device)
    with torch.no_grad():
        if save_with_metadata:
            model.encode(
                batch_feature_dict,
                save_embeddings=True,
                embedding_output_dir=str(output_dir),
                sample_names=names,
                feature_key="h_V_last_layer",
            )
        else:
            h_v, _, _ = model.encode(batch_feature_dict, save_embeddings=False)
            save_minimal_embeddings(h_v, batch_feature_dict, output_dir, names)


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, atom_context_num = load_model(args, device)

    pdb_files = sorted(args.pdb_dir.glob("*.pdb"))
    if args.limit and args.limit > 0:
        pdb_files = pdb_files[: args.limit]
    total_before_shard = len(pdb_files)
    pdb_files = shard_files(pdb_files, args.num_shards, args.shard_index)

    if args.sort_by_length:
        pdb_files = sorted(pdb_files, key=estimate_length_from_pdb)

    print(f"found {total_before_shard} pdb files before sharding")
    print(f"processing {len(pdb_files)} pdb files on shard {args.shard_index}/{args.num_shards}")
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"batch_size: {args.batch_size}")
    print(f"save_mode: {'with_metadata' if args.save_with_metadata else 'minimal_h_V_only'}")

    batch_items = []
    processed = 0
    for pdb_path in pdb_files:
        out_path = output_dir / f"{pdb_path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            continue

        try:
            protein_dict, _, _, _, _ = parse_PDB(
                str(pdb_path),
                device=torch.device("cpu"),
                chains=[],
                parse_all_atoms=args.parse_all_atoms or bool(args.ligand_mpnn_use_side_chain_context),
                parse_atoms_with_zero_occupancy=False,
            )

            protein_dict["chain_mask"] = torch.ones_like(protein_dict["S"], device=protein_dict["S"].device)

            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=args.cutoff_for_score,
                use_atom_context=args.use_atom_context,
                number_of_ligand_atoms=atom_context_num,
                model_type="ligand_mpnn",
            )
            feature_dict["name"] = pdb_path.stem
            feature_dict["source_path"] = str(pdb_path)
            batch_items.append(feature_dict)

            if len(batch_items) >= args.batch_size:
                process_batch(model, batch_items, output_dir, device, args.save_with_metadata)
                processed += len(batch_items)
                batch_items = []
                if processed % 100 == 0 or processed == len(pdb_files):
                    print(f"[{processed}/{len(pdb_files)}] shard_done")
        except Exception as exc:
            print(f"[FAILED] {pdb_path.name}: {exc}")

    if batch_items:
        process_batch(model, batch_items, output_dir, device, args.save_with_metadata)
        processed += len(batch_items)

    print(f"[{processed}/{len(pdb_files)}] shard_done")
    print("embedding extraction finished")


if __name__ == "__main__":
    main()
