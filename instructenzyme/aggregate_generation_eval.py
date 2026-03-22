#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate shard-wise generation-eval outputs.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_records", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_paths = sorted(args.input_dir.glob("shard-*.jsonl"))
    if not record_paths:
        raise FileNotFoundError(f"No shard-*.jsonl files found under {args.input_dir}")

    count = 0
    total_native_len = 0
    total_generated_len = 0
    total_matches = 0
    exact_match_count = 0
    stop_count = 0
    per_seq_recovery_sum = 0.0
    length_ratio_sum = 0.0

    if args.output_records is not None:
        args.output_records.parent.mkdir(parents=True, exist_ok=True)
        merged_f = args.output_records.open("w")
    else:
        merged_f = None

    try:
        for path in record_paths:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    count += 1
                    total_native_len += int(record["native_length"])
                    total_generated_len += int(record["generated_length"])
                    total_matches += int(record["matches"])
                    exact_match_count += int(bool(record["exact_match"]))
                    stop_count += int(bool(record["ended_with_stop"]))
                    per_seq_recovery_sum += float(record["recovery"])
                    length_ratio_sum += float(record["length_ratio"])
                    if merged_f is not None:
                        merged_f.write(json.dumps(record) + "\n")
    finally:
        if merged_f is not None:
            merged_f.close()

    summary = {
        "count": count,
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
        "input_dir": str(args.input_dir.resolve()),
        "record_paths": [str(p.resolve()) for p in record_paths],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
