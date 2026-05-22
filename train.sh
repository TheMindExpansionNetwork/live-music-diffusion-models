# Example training launch script.
# Edit the paths below to point at your own config / checkpoint / output locations.

# Repo root (defaults to the directory containing this script).
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model and dataset configs.
export MODEL_CONFIG="$REPO_ROOT/stable_audio_tools/configs/model_configs/txt2audio/saos_self_forcing_arc_long.json"
export DATASET_CONFIG="$REPO_ROOT/stable_audio_tools/configs/dataset_configs/local_training_example.json"

# Training defaults and (optional) pretrained checkpoint to initialize from.
export SCRIPT_CONFIG="$REPO_ROOT/defaults.ini"
export PRETRAINED_CKPT_PATH="/path/to/pretrained.ckpt"

# Where to write checkpoints.
export SAVE_DIR="./checkpoints"

export ENABLE_TORCH_COMPILE="1"
export USE_CHECKPOINTING="0"

CUDA_VISIBLE_DEVICES=0,1 python train.py \
        --model-config "$MODEL_CONFIG" \
        --dataset-config "$DATASET_CONFIG" \
        --config-file "$SCRIPT_CONFIG" \
        --pretrained-ckpt-path "$PRETRAINED_CKPT_PATH" \
        --save-dir "$SAVE_DIR" \
        --num-workers 48 --accum-batches 1 \
        --batch-size 40 --checkpoint-every 2000 --precision 16-mixed --name "experiment"
