#!/bin/bash

SCRIPT_PATH="scaling_data.py"
SAVE_DIR=""
REPEATS=3
MODELS=("state-spaces/mamba-1.4b" "EleutherAI/pythia-1.4b")
EXPTS=(1 2)
DTYPES=("float32" "float16")
MODEL_SPECIFIED=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -s|--save-dir)
            SAVE_DIR="$2"
            shift
            shift
            ;;
        -r|--repeats)
            REPEATS="$2"
            shift
            shift
            ;;
        -m|--model)
            MODELS=("$2")
            MODEL_SPECIFIED=true
            shift
            shift
            ;;
        --script-path)
            SCRIPT_PATH="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$SAVE_DIR" ]]; then
    echo "Error: Save directory not provided"
    exit 1
fi

echo "Save directory: $SAVE_DIR"
echo "Repeats: $REPEATS"
echo "Script path: $SCRIPT_PATH"

if ! $MODEL_SPECIFIED; then
    echo "Running benchmarks for default models:"
else
    echo "Running benchmarks for specified model: ${MODELS[0]}"
fi

mkdir -p "$SAVE_DIR"

for model in "${MODELS[@]}"; do
    for expt in "${EXPTS[@]}"; do
        for dtype in "${DTYPES[@]}"; do
            echo "Running benchmark: Model=$model, Experiment=$expt, Dtype=$dtype"
            python $SCRIPT_PATH --model-name "$model" \
                                --save-directory "$SAVE_DIR" \
                                --expt $expt \
                                --dtype "$dtype" \
                                --repeats "$REPEATS"
        done
    done
done
