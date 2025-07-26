#!/bin/bash

# Set which GPUs to use (e.g., "0,1" or "0,1,2,3").
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Exit on any error
set -e

# --- Training Configuration ---
OUTPUT_DIR="./output_dir"
DATASET_PATH="./dataset"
BATCH_SIZE=32
EPOCHS=10
PATIENCE=5
NUM_WORKERS=4
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=1
CLIP_GRAD=1.0
PROJECT_NAME="CIFAR10-Training-Accelerate-WandB"
RUN_NAME="SimpleCNN-CIFAR10"
# --- End of Training Configuration ---

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Function to print colored output
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if accelerate is installed
if ! python -c "import accelerate" &> /dev/null; then
    print_error "Accelerate is not installed. Please install it with: pip install accelerate"
    exit 1
fi

# Automatically determine the number of GPUs from CUDA_VISIBLE_DEVICES
# or all available GPUs if the variable is not set.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Parse CUDA_VISIBLE_DEVICES to count specified GPUs
    N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    echo "Using $N_GPUS GPUs from CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
    # Use all available GPUs if CUDA_VISIBLE_DEVICES is not set
    N_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
    echo "Using all $N_GPUS available GPUs"
fi

# Set WandB mode (online, offline, or disabled)
export WANDB_MODE=online

# Set Python path to current directory
export PYTHONPATH="${PYTHONPATH}:."

# Run training using accelerate launch
print_info "Starting CIFAR-10 training with Accelerate..."
print_info "Launching training with the following configuration:"
print_info "  - GPUs: $CUDA_VISIBLE_DEVICES"
print_info "  - Num Processes: $N_GPUS"
print_info "  - Output directory: ${OUTPUT_DIR}"
print_info "  - Dataset path: ${DATASET_PATH}"
print_info "  - Batch size: ${BATCH_SIZE}"
print_info "  - Epochs: ${EPOCHS}"
print_info "  - Learning rate: ${LEARNING_RATE}"
print_info "  - Weight decay: ${WEIGHT_DECAY}"
print_info "  - Warmup epochs: ${WARMUP_EPOCHS}"

accelerate launch --num_processes=$N_GPUS main.py \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_path "${DATASET_PATH}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --epoch "${EPOCHS}" \
    --lr "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --patience "${PATIENCE}" \
    --clip_grad "${CLIP_GRAD}" \
    --project_name "${PROJECT_NAME}" \
    --run_name "${RUN_NAME}" || {
    print_error "Training failed!"
    exit 1
}

print_success "Training completed successfully!"
print_info "Results saved in: ${OUTPUT_DIR}" 