#!/bin/bash

# Set which GPUs to use (e.g., "0,1" or "0,1,2,3").
# The script will automatically count them to set the number of processes for torchrun.
# If this variable is not set, torch will use all available GPUs.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Exit on any error
set -e

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

print_info "Starting CIFAR-10 training with Accelerate..."

# Set WandB mode (online, offline, or disabled)
# For better automation, set WANDB_API_KEY environment variable
export WANDB_MODE=online

# --- Directory config ---
OUTPUT_DIR="./output_dir"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Set Python path to current directory
export PYTHONPATH="${PYTHONPATH}:."

# Run training using accelerate launch with proper error handling
print_info "Launching training with the following configuration:"
print_info "  - GPUs: $CUDA_VISIBLE_DEVICES"
print_info "  - Num Processes: $N_GPUS"
print_info "  - Output directory: ${OUTPUT_DIR}"
print_info "  - Dataset path: ./dataset"
print_info "  - Batch size: 32"
print_info "  - Epochs: 100"
print_info "  - Learning rate: 1e-3"

accelerate launch --num_processes=$N_GPUS main.py \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_path ./dataset \
    --num_workers 4 \
    --batch_size 32 \
    --epoch 5 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --warmup_epochs 1 \
    --patience 20 \
    --clip_grad 1.0 \
    --project_name "CIFAR10-Training-Accelerate" \
    --run_name "SimpleCNN-CIFAR10" || {
    print_error "Training failed!"
    exit 1
}

print_success "Training completed successfully!"
print_info "Results saved in: ${OUTPUT_DIR}" 