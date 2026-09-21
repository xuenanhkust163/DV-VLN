#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${DVVLN_PRETRAINED_PATH:?Set DVVLN_PRETRAINED_PATH to the LLaMA checkpoint directory}"
: "${DVVLN_TOKENIZER_PATH:?Set DVVLN_TOKENIZER_PATH to tokenizer.model}"

DATA_ROOT="${DVVLN_DATA_ROOT:-$REPO_ROOT/datasets}"
CAPTION_DIR="${DVVLN_CAPTION_DIR:-$DATA_ROOT/R2R/captions}"
OUTPUT_DIR="${DVVLN_OUTPUT_DIR:-$DATA_ROOT/R2R/exprs/dv-vln/rxr}"
GPU="${DVVLN_GPU:-0}"
SEED="${DVVLN_SEED:-0}"
MASTER_PORT="${DVVLN_MASTER_PORT:-12229}"

cd "$REPO_ROOT/finetune_src"
CUDA_VISIBLE_DEVICES="$GPU" torchrun \
  --master_port "$MASTER_PORT" --nproc_per_node 1 r2r/main.py \
  --root_dir "$DATA_ROOT" \
  --candidate_cap_dir "$CAPTION_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset rxr --only_en --vlnbert cmt --ob_type cand \
  --world_size 1 --seed "$SEED" \
  --num_l_layers 9 --num_x_layers 4 \
  --hist_enc_pano --hist_pano_num_layers 2 \
  --fix_lang_embedding --fix_hist_embedding \
  --features vit-16-ori --feedback sample \
  --max_action_len 20 --max_instr_len 250 --max_seq_len 1024 \
  --image_feat_size 512 --angle_feat_size 4 \
  --batch_size 32 --feat_dropout 0.4 --dropout 0.5 \
  --llm_predict --stop_first \
  --pretrained_path "$DVVLN_PRETRAINED_PATH" \
  --tokenizer_path "$DVVLN_TOKENIZER_PATH" \
  --llama_type llama_peft --dtype fp16 \
  --verification_mode "${DVVLN_VERIFICATION_MODE:-dual}" \
  --num_samples "${DVVLN_NUM_SAMPLES:-4}" \
  --verification_attempts "${DVVLN_VERIFICATION_ATTEMPTS:-4}" \
  --top_p "${DVVLN_TOP_P:-0.9}" \
  --test --submit "$@"
