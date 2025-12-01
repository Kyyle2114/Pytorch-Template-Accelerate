#!/bin/bash

# Set which GPUs to use (e.g., "0,1" or "0,1,2,3").
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Exit on any error
set -e

# Function to print colored output
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

# Automatically determine the number of GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    echo "Using $N_GPUS GPUs from CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
    N_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
    echo "Using all $N_GPUS available GPUs"
fi

# --- Configuration ---
CONFIG_FILE="config/default.yaml"

# Run training
print_info "Starting training with config: ${CONFIG_FILE}"
print_info "GPUs: $CUDA_VISIBLE_DEVICES (${N_GPUS} processes)"

accelerate launch --num_processes=$N_GPUS --main_process_port 0 main.py \
    --config "${CONFIG_FILE}" >> train.log 2>&1

print_info "Training completed successfully!"