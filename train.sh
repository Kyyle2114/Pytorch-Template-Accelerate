#!/bin/bash

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

print_info "Starting CIFAR-10 training with Accelerate..."

# Set which GPUs to use (e.g., "0,1" or "0,1,2,3").
# Accelerate will automatically detect and use the available GPUs.
export CUDA_VISIBLE_DEVICES=0,1

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
print_info "  - Output directory: ${OUTPUT_DIR}"
print_info "  - Dataset path: ./dataset"
print_info "  - Batch size: 32"
print_info "  - Epochs: 100"
print_info "  - Learning rate: 1e-3"

accelerate launch main.py \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_path ./dataset \
    --num_workers 4 \
    --batch_size 32 \
    --epoch 100 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --patience 20 \
    --clip_grad 1.0 \
    --project_name "CIFAR10-Training-Accelerate" \
    --run_name "SimpleCNN-CIFAR10" || {
    print_error "Training failed!"
    exit 1
}

print_success "Training completed successfully!"
print_info "Results saved in: ${OUTPUT_DIR}" 