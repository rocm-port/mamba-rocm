#!/bin/bash

SCRIPT_PATH="/home/ajassani/hipify_torch/mamba-rocm/benchmarks/scaling_data.py"
SAVE_DIR="/home/ajassani/hipify_torch/data_collected/april19_2024_causal_conv"
REPEATS=3
MODELS=("state-spaces/mamba-1.4b" "EleutherAI/pythia-1.4b")
EXPTS=(1 2)
DTYPES=("float32" "float16")

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
