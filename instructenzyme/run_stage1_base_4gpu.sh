#!/usr/bin/env bash
set -euo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate instructenzyme

export PYTHONPATH=/home/ubuntu/cqr_files/protein_design:/home/ubuntu/cqr_files/protein_design/LLaVA${PYTHONPATH:+:$PYTHONPATH}
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL_PATH=${1:-/home/ubuntu/cqr_files/protein_design/progen2-base}
TRAIN_INDEX=${2:-/home/ubuntu/cqr_files/protein_design/instructenzyme/data/index/train.jsonl}
VAL_INDEX=${3:-/home/ubuntu/cqr_files/protein_design/instructenzyme/data/index/val.jsonl}
OUTPUT_DIR=${4:-/home/ubuntu/cqr_files/protein_design/instructenzyme/runs/progen2-base-stage1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1000}
BATCH_SIZE=${BATCH_SIZE:-2}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-2e-4}
EVAL_EVERY=${EVAL_EVERY:-100}
SAVE_EVERY=${SAVE_EVERY:-100}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-512}

mkdir -p "${OUTPUT_DIR}"

exec torchrun --standalone --nproc_per_node=4 /home/ubuntu/cqr_files/protein_design/instructenzyme/train_stage1.py \
  --model_name_or_path "${MODEL_PATH}" \
  --train_index "${TRAIN_INDEX}" \
  --val_index "${VAL_INDEX}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --max_val_samples "${MAX_VAL_SAMPLES}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --bf16
