#!/bin/bash
# Multi-model comparison eval runner
# Runs the inspect-ai reconstruction task across multiple models and embedding configs
# Usage: ./run_multi_model_eval.sh [--limit N] [--instruct MODE] [--pos POS]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
LIMIT=""
INSTRUCT="basevocab"
POS="ADV"
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit) LIMIT="-T limit=$2"; shift 2 ;;
        --instruct) INSTRUCT="$2"; shift 2 ;;
        --pos) POS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Ensure venv
if [ ! -d .venv ]; then
    echo "ERROR: .venv not found. Run: uv sync --extra eval"
    exit 1
fi
source .venv/bin/activate

# Ensure FST is built
if [ ! -f ../fst/fst/build/cg3/validator.bin ]; then
    echo "Building FST/CG3 artifacts..."
    make -C ../fst cg3-check
fi

# Embedding configs to test
declare -a CONFIGS=(
    "hf-voyage:env.hf-voyage.sh:openai/gpt-oss-120b:cheapest"
    "hf-model2vec:env.hf-model2vec.sh:openai/gpt-oss-120b:cheapest"
)

# Create results directory
RESULTS_DIR="evals/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/eval_run.log"

echo "======================================"
echo "Multi-Model Eval Run"
echo "======================================"
echo "Instruct: $INSTRUCT"
echo "POS filter: $POS"
echo "Results dir: $RESULTS_DIR"
echo "Log: $LOG_FILE"
echo ""

# Run each config
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r CONFIG_NAME ENV_FILE MODEL <<< "$config"

    echo "======================================" | tee -a "$LOG_FILE"
    echo "Testing: $CONFIG_NAME (model: $MODEL)" | tee -a "$LOG_FILE"
    echo "======================================" | tee -a "$LOG_FILE"

    # Source the environment
    if [ ! -f "$ENV_FILE" ]; then
        echo "WARNING: $ENV_FILE not found, skipping $CONFIG_NAME" | tee -a "$LOG_FILE"
        continue
    fi
    source "$ENV_FILE"

    # Run the eval
    export INSPECT_EVAL_MODEL="hf-inference-providers/$MODEL"
    OUTPUT_DIR="$RESULTS_DIR/$CONFIG_NAME"

    echo "Output dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "Model: $INSPECT_EVAL_MODEL" | tee -a "$LOG_FILE"
    echo "Starting eval at $(date)" | tee -a "$LOG_FILE"

    if inspect eval evals/reconstruction.py \
        -T "instruct=$INSTRUCT" \
        -T "pos=$POS" \
        $LIMIT \
        --model "$INSPECT_EVAL_MODEL" \
        --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"; then

        echo "✓ Eval completed successfully for $CONFIG_NAME" | tee -a "$LOG_FILE"

        # Extract metrics summary
        if [ -f "$OUTPUT_DIR/results.json" ]; then
            python3 << 'PYSCRIPT'
import json
import sys
results_file = sys.argv[1]
with open(results_file) as f:
    data = json.load(f)
    if 'results' in data:
        results = data['results']
        if isinstance(results, list) and results:
            metrics = results[0].get('metrics', {})
            print("\nMetrics Summary:")
            for k, v in sorted(metrics.items()):
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
PYSCRIPT
            python3 -c "
import json
with open('$OUTPUT_DIR/results.json') as f:
    data = json.load(f)
    if 'results' in data:
        results = data['results']
        if isinstance(results, list) and results:
            metrics = results[0].get('metrics', {})
            print('\nMetrics Summary:')
            for k, v in sorted(metrics.items()):
                if isinstance(v, (int, float)):
                    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
" 2>/dev/null || true
        fi
    else
        echo "✗ Eval failed for $CONFIG_NAME" | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"
    sleep 2
done

echo "======================================"
echo "Eval run complete!"
echo "Results saved to: $RESULTS_DIR"
echo "View results: inspect view $RESULTS_DIR"
echo "======================================"
