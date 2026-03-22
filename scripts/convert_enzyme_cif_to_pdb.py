#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from Bio.PDB import MMCIFParser


_ALLOWED_ALTLOCS = {" ", "A", "1"}


def sanitize_resname(resname: str) -> str:
    resname = (resname or "UNK").strip().upper()
    if len(resname) <= 3:
        return resname.rjust(3)
    # PDB resname field is 3 columns; truncate long ligand names deterministically.
    return resname[:3]


def sanitize_chain_id(chain_id: str) -> str:
    chain_id = (chain_id or "A").strip()
    return chain_id[0] if chain_id else "A"


def format_pdb_atom_line(record_name: str, serial: int, atom, residue, chain_id: str, resseq: int, icode: str) -> str:
    name = atom.get_fullname()
    if len(name) < 4:
        name = f"{name:>4}"
    name = name[:4]

    altloc = atom.get_altloc()
    altloc = altloc if altloc in _ALLOWED_ALTLOCS else " "

    resname = sanitize_resname(residue.get_resname())
    chain_id = sanitize_chain_id(chain_id)
    icode = (icode or " ").strip() or " "

    x, y, z = atom.get_coord()
    occupancy = atom.get_occupancy()
    bfactor = atom.get_bfactor()
    occupancy = 1.00 if occupancy is None else occupancy
    bfactor = 0.00 if bfactor is None else bfactor

    element = (atom.element or "").strip().upper()
    if not element:
        stripped = atom.get_name().strip()
        element = stripped[0].upper() if stripped else ""
    element = element[:2].rjust(2)

    charge = "  "

    return (
        f"{record_name:<6}{serial:>5d} {name}{altloc}{resname:>3} {chain_id}"
        f"{resseq:>4d}{icode}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{occupancy:>6.2f}{bfactor:>6.2f}          "
        f"{element}{charge}"
    )


def write_structure_to_pdb(structure, output_path: Path) -> None:
    lines = []
    serial = 1
    models = list(structure)
    multi_model = len(models) > 1

    for model_index, model in enumerate(models, start=1):
        if multi_model:
            lines.append(f"MODEL     {model_index:>4d}")

        for chain in model:
            chain_id = sanitize_chain_id(chain.id)
            for residue in chain:
                hetflag, resseq, icode = residue.id
                if hetflag == "W":
                    continue

                record_name = "ATOM"
                if str(hetflag).strip() and str(hetflag).strip() != "W":
                    record_name = "HETATM"

                for atom in residue:
                    altloc = atom.get_altloc()
                    if altloc not in _ALLOWED_ALTLOCS:
                        continue
                    line = format_pdb_atom_line(
                        record_name=record_name,
                        serial=serial,
                        atom=atom,
                        residue=residue,
                        chain_id=chain_id,
                        resseq=int(resseq),
                        icode=icode,
                    )
                    lines.append(line)
                    serial += 1

            lines.append(f"TER   {serial:>5d}      {sanitize_resname(residue.get_resname())} {chain_id}{int(resseq):>4d}{((icode or ' ').strip() or ' ')}")
            serial += 1

        if multi_model:
            lines.append("ENDMDL")

    lines.append("END")
    output_path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert enzyme_data mmCIF files to valid fixed-width PDB while keeping protein and ligand atoms."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/home/ubuntu/cqr_files/protein_design/enzyme_data"),
        help="Directory containing input .cif files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/ubuntu/cqr_files/protein_design/enzyme_pdb_test"),
        help="Directory to write output .pdb files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Only convert the first N CIF files. Use 0 or a negative value for all files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PDB files in the output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = MMCIFParser(QUIET=True)

    cif_files = sorted(input_dir.glob("*.cif"))
    if args.limit and args.limit > 0:
        cif_files = cif_files[: args.limit]

    print(f"found {len(cif_files)} cif files")
    print("long ligand residue names will be truncated to 3 characters for valid PDB output")

    for i, cif_path in enumerate(cif_files, 1):
        out_path = output_dir / f"{cif_path.stem}.pdb"
        if out_path.exists() and not args.overwrite:
            continue

        structure = parser.get_structure(cif_path.stem, str(cif_path))
        write_structure_to_pdb(structure, out_path)

        if i % 100 == 0 or i == len(cif_files):
            print(f"[{i}/{len(cif_files)}] done")

    print("conversion finished")


if __name__ == "__main__":
    main()
