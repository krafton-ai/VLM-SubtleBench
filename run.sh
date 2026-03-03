#!/bin/bash

# Evaluation script for the unified VLM-SubtleBench dataset
# Supports filtering by category and domain

# Check if running with bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script requires bash. Please run with: bash run.sh"
    exit 1
fi

# Default values
DEFAULT_MODEL="gpt-4o"
DEFAULT_PROMPT_TYPE="no-reasoning"
MAX_QUESTIONS=""

# Parse command line arguments
MODEL="$DEFAULT_MODEL"
PROMPT_TYPE="$DEFAULT_PROMPT_TYPE"
CATEGORY=""
DOMAIN=""
SPLIT=""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model MODEL           Set the model name (default: $DEFAULT_MODEL)"
    echo "  --prompt_type TYPE      Set the prompt type (default: $DEFAULT_PROMPT_TYPE)"
    echo "  --max_questions N       Limit number of questions (default: all)"
    echo "  --split SPLIT           Filter by split: test, val, or all (default: test via config)"
    echo "  --category CATEGORY     Filter by category (default: all)"
    echo "  --domain DOMAIN         Filter by domain (default: all)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Available models: gpt-4o, o3, gpt-5, gemini-2.5-flash, anthropic/claude-sonnet-4, etc."
    echo "Available prompt types: standard, no_reasoning, concatenated, grid, overlapped, camera_augmented"
    echo "Categories: action, attribute, emotion, existence, quality, quantity, spatial, state, temporal, viewpoint"
    echo "Domains: natural, industrial, medical, aerial, synthetic"
    echo ""
    echo "Examples:"
    echo "  $0                                              # Evaluate all items"
    echo "  $0 --model gpt-4o --category attribute          # Only attribute category"
    echo "  $0 --model gemini-2.5-flash --domain medical    # Only medical domain"
    echo "  $0 --max_questions 100 --category state         # 100 state questions"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --prompt_type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --max_questions)
            MAX_QUESTIONS="$2"
            shift 2
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

# Validate that required values are set
if [ -z "$MODEL" ]; then
    echo "Model cannot be empty"
    exit 1
fi

if [ -z "$PROMPT_TYPE" ]; then
    echo "Prompt type cannot be empty"
    exit 1
fi

# Generate log file name with incremental numbering
generate_log_filename() {
    mkdir -p eval_logs

    local base_name="${PROMPT_TYPE}-${MODEL}"
    base_name=$(echo "$base_name" | sed 's|/|-|g')

    local counter=1
    local log_file="eval_logs/${base_name}-${counter}.log"

    while [ -f "$log_file" ]; do
        counter=$((counter + 1))
        log_file="eval_logs/${base_name}-${counter}.log"
    done

    echo "$log_file"
}

LOG_FILE=$(generate_log_filename)

echo "Starting evaluation"
echo "Started at: $(date)"
echo "Model: $MODEL"
echo "Prompt type: $PROMPT_TYPE"
echo "Split: ${SPLIT:-test (default)}"
echo "Category: ${CATEGORY:-all}"
echo "Domain: ${DOMAIN:-all}"
echo "Max questions: ${MAX_QUESTIONS:-all}"
echo "Log file: $LOG_FILE"
echo ""

# Start logging everything to file
exec > >(tee -a "$LOG_FILE") 2>&1

# Build CLI overrides
CLI_ARGS=(
    "model.llm_name=$MODEL"
    "model.prompt_type=$PROMPT_TYPE"
    "model.use_multithreading=true"
    "model.max_workers=16"
)

if [ -n "$MAX_QUESTIONS" ]; then
    CLI_ARGS+=("data.max_questions=$MAX_QUESTIONS")
fi

if [ -n "$CATEGORY" ]; then
    CLI_ARGS+=("data.category=$CATEGORY")
fi

if [ -n "$DOMAIN" ]; then
    CLI_ARGS+=("data.domain=$DOMAIN")
fi

if [ -n "$SPLIT" ]; then
    if [ "$SPLIT" = "all" ]; then
        CLI_ARGS+=("data.split=null")
    else
        CLI_ARGS+=("data.split=$SPLIT")
    fi
fi

# Run evaluation
python scripts/evaluate_multiple_choice.py "${CLI_ARGS[@]}" 2>&1

exit_code=$?

echo ""
echo "Finished at: $(date)"
echo "Log saved to: $LOG_FILE"

if [ $exit_code -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation failed with exit code $exit_code"
fi

exit $exit_code
