export CUDA_VISIBLE_DEVICES=1

PROJECT_ROOT=/path/to/CalibraRL

INPUT_FILE="$PROJECT_ROOT/datasets/llama3/tqa/test/test.json"
BASE_MODEL=/path/to/models/Llama-3-8B-Instruct
LORA_MODEL=/path/to/lora/checkpoint
MAX_TOKENS=8

python "$PROJECT_ROOT/bash/inference.py" \
    --input "$INPUT_FILE" \
    --base_model "$BASE_MODEL" \
    --lora_model "$LORA_MODEL" \
    --max_tokens "$MAX_TOKENS"