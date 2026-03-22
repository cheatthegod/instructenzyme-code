#!/usr/bin/env bash
set -euo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate instructenzyme

export PYTHONPATH=/home/ubuntu/cqr_files/protein_design:/home/ubuntu/cqr_files/protein_design/LLaVA${PYTHONPATH:+:$PYTHONPATH}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODEL_PATH=${1:-/home/ubuntu/cqr_files/protein_design/progen2-base}
PROJECTOR_CKPT=${2:-/home/ubuntu/cqr_files/protein_design/instructenzyme/runs/progen2-base-stage1-1k/best/projector.pt}
INDEX_PATH=${3:-/home/ubuntu/cqr_files/protein_design/instructenzyme/data/index/test.jsonl}
OUTPUT_DIR=${4:-/home/ubuntu/cqr_files/protein_design/instructenzyme/generation_eval/progen2-base-stage1-best-test-greedy-batched}
MAX_SAMPLES=${MAX_SAMPLES:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-0}
NUM_SHARDS=${NUM_SHARDS:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
DO_SAMPLE=${DO_SAMPLE:-0}

mkdir -p "${OUTPUT_DIR}/logs"

for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU=${SHARD}
  echo "Launching generation shard ${SHARD}/${NUM_SHARDS} on GPU ${GPU} with batch_size=${BATCH_SIZE}"
  CUDA_VISIBLE_DEVICES=${GPU} python /home/ubuntu/cqr_files/protein_design/instructenzyme/generate_stage1.py \
    --model_name_or_path "${MODEL_PATH}" \
    --projector_ckpt "${PROJECTOR_CKPT}" \
    --index_path "${INDEX_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${SHARD}" \
    --batch_size "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --bf16 \
    $( [ "${DO_SAMPLE}" = "1" ] && echo "--do_sample" ) \
    > "${OUTPUT_DIR}/logs/gpu_${GPU}.log" 2>&1 &
done

wait

echo "All generation shards finished, aggregating..."
python /home/ubuntu/cqr_files/protein_design/instructenzyme/aggregate_generation_eval.py \
  --input_dir "${OUTPUT_DIR}" \
  --output_json "${OUTPUT_DIR}/summary.json" \
  --output_records "${OUTPUT_DIR}/all_records.jsonl"
