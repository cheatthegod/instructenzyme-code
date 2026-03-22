#!/usr/bin/env bash
set -euo pipefail

PDB_DIR=${1:-/home/ubuntu/cqr_files/protein_design/enzyme_pdb}
OUT_DIR=${2:-/home/ubuntu/cqr_files/protein_design/ligandmpnn_emb}
CKPT=${3:-/home/ubuntu/cqr_files/protein_design/LigandMPNN/model_params/ligandmpnn_v_32_005_25.pt}
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${LOG_DIR:-${OUT_DIR}/logs}
mkdir -p "$OUT_DIR" "$LOG_DIR"

for ((i=0; i<NUM_GPUS; i++)); do
  echo "Launching shard $i/$NUM_GPUS on GPU $i with batch_size=$BATCH_SIZE"
  CUDA_VISIBLE_DEVICES=$i \
    python "$SCRIPT_DIR/extract_ligandmpnn_embeddings.py" \
      --pdb_dir "$PDB_DIR" \
      --output_dir "$OUT_DIR" \
      --checkpoint "$CKPT" \
      --device cuda \
      --num_shards "$NUM_GPUS" \
      --shard_index "$i" \
      --batch_size "$BATCH_SIZE" \
      --sort_by_length \
      > "$LOG_DIR/gpu_${i}.log" 2>&1 &
done

wait

echo "all shards finished"
