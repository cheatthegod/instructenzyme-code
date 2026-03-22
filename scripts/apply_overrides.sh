#!/usr/bin/env bash
set -euo pipefail

EXPORT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_ROOT=${1:-$(pwd)}

for required in "${TARGET_ROOT}/LLaVA" "${TARGET_ROOT}/LigandMPNN"; do
  if [[ ! -d "${required}" ]]; then
    echo "Missing required upstream directory: ${required}" >&2
    exit 1
  fi
done

cp -f "${EXPORT_ROOT}/third_party_overrides/LigandMPNN/model_utils.py" "${TARGET_ROOT}/LigandMPNN/model_utils.py"
cp -f "${EXPORT_ROOT}/third_party_overrides/LigandMPNN/run.py" "${TARGET_ROOT}/LigandMPNN/run.py"
cp -f "${EXPORT_ROOT}/third_party_overrides/LigandMPNN/score.py" "${TARGET_ROOT}/LigandMPNN/score.py"
cp -f "${EXPORT_ROOT}/third_party_overrides/LigandMPNN/extract_ligandmpnn_embeddings.py" "${TARGET_ROOT}/LigandMPNN/extract_ligandmpnn_embeddings.py"
cp -f "${EXPORT_ROOT}/third_party_overrides/LigandMPNN/run_extract_ligandmpnn_embeddings_4gpu.sh" "${TARGET_ROOT}/LigandMPNN/run_extract_ligandmpnn_embeddings_4gpu.sh"

cp -f "${EXPORT_ROOT}/third_party_overrides/LLaVA/llava/model/multimodal_encoder/dummy_encoder.py" "${TARGET_ROOT}/LLaVA/llava/model/multimodal_encoder/dummy_encoder.py"
cp -f "${EXPORT_ROOT}/third_party_overrides/LLaVA/llava/model/multimodal_encoder/embedding_dataset.py" "${TARGET_ROOT}/LLaVA/llava/model/multimodal_encoder/embedding_dataset.py"
cp -f "${EXPORT_ROOT}/third_party_overrides/LLaVA/llava/model/multimodal_encoder/builder.py" "${TARGET_ROOT}/LLaVA/llava/model/multimodal_encoder/builder.py"
cp -f "${EXPORT_ROOT}/third_party_overrides/LLaVA/llava/model/multimodal_projector/builder.py" "${TARGET_ROOT}/LLaVA/llava/model/multimodal_projector/builder.py"
cp -f "${EXPORT_ROOT}/third_party_overrides/LLaVA/llava/model/llava_arch.py" "${TARGET_ROOT}/LLaVA/llava/model/llava_arch.py"
cp -f "${EXPORT_ROOT}/third_party_overrides/LLaVA/llava/train/train.py" "${TARGET_ROOT}/LLaVA/llava/train/train.py"

echo "Applied InstructEnzyme overrides into ${TARGET_ROOT}"
