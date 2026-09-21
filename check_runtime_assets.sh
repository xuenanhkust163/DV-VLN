#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "DV-VLN runtime asset check"
echo "repo: $ROOT"

for path in \
  "$ROOT/datasets/R2R/annotations/R2R_train_enc.json" \
  "$ROOT/datasets/R2R/annotations/R2R_val_unseen_subset_enc.json" \
  "$ROOT/datasets/R2R/features/vit-16.hdf5" \
  "$ROOT/datasets/R2R/features/dvae_probs.hdf5" \
  "$ROOT/datasets/R2R/connectivity/scans.txt" \
  "$ROOT/visual_token_count.npy"; do
  if [ -e "$path" ]; then
    printf 'OK   %s\n' "$path"
  else
    printf 'MISS %s\n' "$path"
  fi
done

echo
echo "Model assets are supplied separately via --resume_file and --cfg_name."
echo "LLM mode additionally requires --tokenizer_path and --pretrained_path."
