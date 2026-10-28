#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

DATASET_NUMBER=10000
LEARNING_RATE=1e-6
NUM_EPOCHS=1
BATCH_SIZE=4
EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=8
GROUP_SIZE=32
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=16

TEMPERATURE=1.0
TOP_P=0.95

CLIP_EPS=0.2
KL_COEF=0.01
WARMUP_RATIO=0.03
EVAL_RATIO=0.1

LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0

CONFIDENCE_WEIGHT=0.5

GROUP_REWARD_SCALE=0.5
GROUP_PRIOR_ALPHA=1.0
GROUP_PRIOR_BETA=1.0
GROUP_CONFIDENCE_MODE=bayesian_tail
ANSWERABILITY_THRESHOLD=0.5

SEED=42

MODEL_PATH=/path/to/models/Llama-3-8B-Instruct
PROJECT_ROOT=/path/to/abstention

TRAIN_FILE="$PROJECT_ROOT/src/data_llama3/nq/train.json"
TRAIN_SCRIPT="$PROJECT_ROOT/src/train_calibrarl.py"
OUTPUT_ROOT="$PROJECT_ROOT/saves/saves_raw_brier_reward"
SCRIPT_PATH="$(realpath "$0")"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Error: Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Error: Training data not found: $TRAIN_FILE" >&2
    exit 1
fi

if [[ ! -e "$MODEL_PATH" ]]; then
    echo "Error: Base model not found: $MODEL_PATH" >&2
    exit 1
fi

python -u "$TRAIN_SCRIPT" \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_root "$OUTPUT_ROOT" \
    --dataset_number "$DATASET_NUMBER" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --group_size "$GROUP_SIZE" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --clip_eps "$CLIP_EPS" \
    --kl_coef "$KL_COEF" \
    --warmup_ratio "$WARMUP_RATIO" \
    --eval_ratio "$EVAL_RATIO" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --confidence_weight "$CONFIDENCE_WEIGHT" \
    --group_reward_scale "$GROUP_REWARD_SCALE" \
    --group_prior_alpha "$GROUP_PRIOR_ALPHA" \
    --group_prior_beta "$GROUP_PRIOR_BETA" \
    --group_confidence_mode "$GROUP_CONFIDENCE_MODE" \
    --answerability_threshold "$ANSWERABILITY_THRESHOLD" \
    --seed "$SEED" \
    --script_path "$SCRIPT_PATH" \
    --use_peft \
    --no_fit_confidence_calibration \
    --no_gradient_checkpointing \
    --reward_debug