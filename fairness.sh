#!/bin/bash

MODEL_NAME=${1:-"meta-llama/Llama-3.2-7B"}
OUTPUT_DIR=${2:-"./full_fairness_results"}
TASK_NAME=${3:-"stereoset_intersentence"}
TASK_PATH="evaluation/fairness"


# --- Script ---

echo "Starting fairness evaluation for model: $MODEL_NAME"
echo "Using task: $TASK_NAME"
echo "Results will be saved to: $OUTPUT_DIR"

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME \
    --tasks $TASK_NAME \
    --include_path $TASK_PATH \
    --output_path $OUTPUT_DIR \
    --log_samples \

echo "Evaluation finished."
echo "Check the directory '$OUTPUT_DIR' for detailed results."

