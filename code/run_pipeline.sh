#!/bin/bash

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
ENV_NAME= "env-jailbreak" # "env-advanced_jailbreak"
PYTHON_VERSION="3.10.19"
PYTHON_SCRIPT="pipeline.py"

# ---------------------------------------------------------------
# HELP MENU
# ---------------------------------------------------------------
show_help() {
    echo "---------------------------------------------------------------"
    echo "      A* Jailbreak Benchmark Pipeline - Automation Script      "
    echo "---------------------------------------------------------------"
    echo "Usage: ./run_pipeline.sh [OPTIONS]"
    echo ""
    echo "Data & Pre-computation Options:"
    echo "  --build-dataset --target-models <name_model>  Generates the condensed Alpaca + JBB dataset"
    echo "  --train-hts all    Trains the HTS Matrices for all models in the array"
    echo "  --train-hts <name> Trains the HTS Matrix for a specific model string"
    echo ""
    echo "Script Management Options:"
    echo "  -h, --help         Show this help message and exit."
    echo "  --install-deps     Create the Conda environment, install dependencies, and exit."
    echo "  --compare          Run ALL strategies and compare them"
    echo "  --test-sys         Run Systemwide GPU tests to make sure pipeline execution is flawless"
    echo ""
    echo "ASR Pipeline (Malicious Prompts):"
    echo "  --infer all        Run ASR inference on ALL models in the array"
    echo "  --infer <name>     Run ASR inference on a specific 'llm model'"
    # echo "  --baseline         Run the vanilla defense"
    echo "  --fuse         Run the FUSE Multi-layered Defense"
    echo "  --smoothing        Run the randomized smoothing defense"
    echo "  --enterprise       Run the Stateful Streaming Interceptor"
    echo "  --dcmd             Run the Dual-Phase Cryptographic Manifold Defense"
    echo "  --none             Run with no defense (Control Group)"
    echo ""
    echo "FPR Pipeline (Benign Prompts / Utility):"
    echo "  --eval-fpr all     Run Utility inference on ALL models in the array"
    echo "  --eval-fpr <name>  Run Utility inference on a specific model string"
    echo ""
    echo "Example Usage:"
    echo "  ./run_pipeline.sh --install-deps"
    echo "  ./run_pipeline.sh --build-dataset --train-hts all"
    echo "  ./run_pipeline.sh --infer all --compare (Exhaustive ASR Benchmarking)"
    echo "  ./run_pipeline.sh --eval-fpr all (Exhaustive Utility/FPR Benchmarking)"
    echo "---------------------------------------------------------------"
}

function generate_master_charts() {
    echo "📊 Generating ASR Master Heatmap (Red)..."
    python visualizer.py --input ./model_responses/global_attack_results.csv --type asr_heatmap
    
    echo "🟩 Generating FPR Utility Heatmap (Green)..."
    python visualizer.py --input ./model_responses/global_benign_results.csv --type fpr_heatmap
}

# ---------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------
BUILD_DATASET=0
TRAIN_HTS_MODE="none"
INSTALL_DEPS=0
PIPELINE_ARGS=()
TRAIN_ARGS=()
NUM_MODELS=0
TEST_SYSTEM=0
PIPELINE_HAS_ERRORS=0
INFER_MODE="none"
EVAL_FPR_MODE="none" # Upgraded to handle 'all' or specific model

mkdir -p logs 

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --install-deps)
            INSTALL_DEPS=1
            shift
            ;;
        --build-dataset)
            BUILD_DATASET=1
            shift
            ;;
        --train-hts)
            if [[ -n "$2" && "$2" != -* ]]; then
                TRAIN_HTS_MODE="$2"
                shift 2
            else
                TRAIN_HTS_MODE="all"
                NUM_MODELS=5
                shift 1
            fi
            ;;
        --infer)
            if [[ -n "$2" && "$2" != -* ]]; then
                INFER_MODE="$2"
                shift 2
            else
                INFER_MODE="all"
                shift 1
            fi
            ;;
        --eval-fpr)
            # Upgraded argument logic to mirror --infer
            if [[ -n "$2" && "$2" != -* ]]; then
                EVAL_FPR_MODE="$2"
                shift 2
            else
                EVAL_FPR_MODE="all"
                shift 1
            fi
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --strategy|--loss_type|--l1_lambda|--l2_lambda)
            TRAIN_ARGS+=("$1" "$2")
            shift 2
            ;;
        --test-sys)
            TEST_SYSTEM=1
            shift
            ;;
        *)
            PIPELINE_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---------------------------------------------------------------
# CONDA & ENVIRONMENT SETUP
# ---------------------------------------------------------------
echo "Checking Conda installation..."
if ! command -v conda &> /dev/null; then
    echo "❌ ERROR: Conda is not installed or not in your PATH. Please install Conda first."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "✅ Environment '$ENV_NAME' found."
else
    echo "⚠️  Environment '$ENV_NAME' not found. Provisioning clean environment now..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed to create Conda environment."
        exit 1
    fi
    
    conda activate "$ENV_NAME"
    echo "📦 Installing core Machine Learning & Hardware dependencies..."
    python -m pip install --upgrade pip
    python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    
    echo "📦 Installing Pipeline & LLM Architecture dependencies..."
    conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y
    python -m pip uninstall -y cmake
    conda install -c conda-forge cmake -y
    python -m pip install vllm==0.6.2 transformers>=4.48.0 huggingface-hub>=0.24.0 --extra-index-url https://download.pytorch.org/whl/cu121 --only-binary pyarrow
    python -m pip install jailbreakbench==1.0.0 datasets>=2.20.0 fschat accelerate --only-binary pyarrow
    
    echo "📦 Installing Data, Visualization & Evaluator dependencies..."
    python -m pip install llama-cpp-python==0.3.16
    python -m pip install pandas==2.3.3 matplotlib==3.10.8 seaborn==0.13.2 psutil==7.2.2 tabulate==0.9.0 scikit-learn
    
    echo "✅ Environment setup complete."
fi

if [ "$INSTALL_DEPS" -eq 1 ]; then
    echo "✅ Setup requested. Exiting without running the pipeline."
    exit 0
fi

if [ "$TEST_SYSTEM" -eq 1 ]; then
    echo "======================================================================"
    echo "🚀 RUNNING SYSTEM HARDWARE DIAGNOSTIC (test_sys.py)"
    echo "======================================================================"
    cd ./test
    python test_sys.py
    cd ../
    TEST_EXIT=$?
    
    if [ $TEST_EXIT -eq 0 ]; then
        echo "✅ Hardware check passed. System is ready for dual-GPU inference."
    else
        echo "❌ Hardware check FAILED. Attempting to force purge ghost processes..."
        exit 1
    fi

    if [ "$INFER_MODE" == "none" ] && [ "$BUILD_DATASET" -eq 0 ] && [ "$TRAIN_HTS_MODE" == "none" ] && [ "$EVAL_FPR_MODE" == "none" ]; then
        echo "✅ Test complete. Exiting as no further tasks were requested."
        exit 0
    fi
fi

# Universal Gate Exit
if [ "$INFER_MODE" == "none" ] && [ "$BUILD_DATASET" -eq 0 ] && [ "$TRAIN_HTS_MODE" == "none" ] && [ "$EVAL_FPR_MODE" == "none" ]; then
    echo "✅ Requested offline tasks complete. No tasks requested. Exiting.."
    exit 0
fi

# ---------------------------------------------------------------
# PIPELINE ENV VARIABLES
# ---------------------------------------------------------------

# you can export your Hugging Face Token globally here
# export HF_TOKEN=your_token

export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export GLOO_TIMEOUT_SECONDS=3600
export NCCL_CLIENT_TIMEOUT=3600
export VLLM_RPC_TIMEOUT=360000
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1
export DISTRIBUTED_TIMEOUT=1200  # Increase to 20 minutes

# Clear shared memory segments (often left behind by vLLM) Uncomment only if you are sure not to stop important processes
rm -rf /dev/shm/* # EXTREME CAUTION
# Clean up any "ghost" workers that didn't die naturally
pkill -f vllm.executor.multiproc_worker_utils

MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "google/gemma-2b"
    # "google/gemma-7b-it"
    "meta-llama/Llama-2-7b-chat-hf"
    "lmsys/vicuna-13b-v1.5"
)

# =========================================================
# SECTION A: DATASET GENERATION
# =========================================================
if [ "$BUILD_DATASET" -eq 1 ]; then
    echo "=========================================="
    echo "📊 GENERATING Adv Bench + ALAPCA + JBB DATASET"
    echo "=========================================="
    python dataset_builder.py "${PIPELINE_ARGS[@]}"
    if [ $? -ne 0 ]; then
        echo "❌ Dataset generation failed. Exiting..."
        exit 1
    fi
    echo "✅ Dataset successfully generated and saved."
fi

# =========================================================
# SECTION B: OFFLINE HTS MATRIX TRAINING
# =========================================================
if [ "$TRAIN_HTS_MODE" != "none" ]; then
    echo "=========================================="
    echo "🛡️ INITIATING OFFLINE HTS MATRIX TRAINING"
    echo "=========================================="
    if [ "$TRAIN_HTS_MODE" == "all" ]; then
        for MODEL in "${MODELS[@]}"; do
            echo "Training HTS Matrix for: $MODEL"
            python train_hts.py --model "$MODEL" "${TRAIN_ARGS[@]}"
            if [ $? -ne 0 ]; then
                echo "❌ Training failed for $MODEL. Exiting..."
                exit 1
            fi
        done
    else
        python train_hts.py --model "$TRAIN_HTS_MODE" "${TRAIN_ARGS[@]}"
        if [ $? -ne 0 ]; then exit 1; fi
    fi
    echo "✅ HTS Matrix successfully compiled."
fi

# =========================================================
# SECTION C: ASR PIPELINE (MALICIOUS PROMPTS)
# =========================================================
if [ "$INFER_MODE" != "none" ]; then
    mkdir -p model_responses

    LOG_FILE="logs/exhaustive_execution.log"
    TERMINAL_LOG="logs/exhaustive_terminal_output.txt"
    > "$LOG_FILE"
    > "$TERMINAL_LOG"
    
    # Redirect all output to log files
    exec > >(tee -a "$LOG_FILE" "$TERMINAL_LOG") 2>&1

    echo "📝 Exhaustive ASR Logging Started at $(date)"
    
    if [ "$INFER_MODE" == "all" ]; then
        rm -f "./model_responses/global_full_results.csv"
        rm -f "./model_responses/jailbroken_full_response.csv"
        MODELS_TO_RUN=("${MODELS[@]}")
    else
        MODELS_TO_RUN=("$INFER_MODE")
    fi

    PIPELINE_HAS_ERRORS=0

    for MODEL in "${MODELS_TO_RUN[@]}"; do
        echo "======================================================================"
        echo " 🎯 EVALUATING ASR FOR MODEL: $MODEL"
        echo "======================================================================"

        python "$PYTHON_SCRIPT" "${PIPELINE_ARGS[@]}" --model "$MODEL" 
        
        EXIT_CODE=$?
        sync
        
        if [ $EXIT_CODE -ne 0 ]; then
            echo "⚠️ WARNING: Model $MODEL crashed (Exit Code: $EXIT_CODE)."
            PIPELINE_HAS_ERRORS=1
        fi
        
        sleep 15 # VRAM Cooldown
    done

    echo "======================================================================"
    echo "✅ All ASR model evaluations complete."
    generate_master_charts 
fi

# ==============================================================================
# SECTION D: FPR PIPELINE (BENIGN PROMPTS / UTILITY)
# ==============================================================================
if [ "$EVAL_FPR_MODE" != "none" ]; then
    mkdir -p model_responses logs

    LOG_FILE="logs/exhaustive_benign_execution.log"
    TERMINAL_LOG="logs/exhaustive_benign_terminal.txt"
    > "$LOG_FILE"
    > "$TERMINAL_LOG"
    
    # Redirect all output to benign log files
    exec > >(tee -a "$LOG_FILE" "$TERMINAL_LOG") 2>&1

    echo "======================================================================"
    echo " ⚖️ EVALUATING FALSE POSITIVE RATES (BENIGN UTILITY)"
    echo "======================================================================"
    echo "📝 Exhaustive FPR Logging Started at $(date)"

    # Only wipe global FRP_Results if we are running a completely new sweep
    if [ "$EVAL_FPR_MODE" == "all" ]; then
        rm -f "./model_responses/global_benign_results.csv"
        rm -f "./model_responses/hts_benign_execution.log"
        rm -f "./model_responses/prf_benign_execution.log"
        MODELS_FOR_UTILITY=("${MODELS[@]}")
    else
        MODELS_FOR_UTILITY=("$EVAL_FPR_MODE")
    fi

    FPR_HAS_ERRORS=0

    for MODEL_NAME in "${MODELS_FOR_UTILITY[@]}"; do
        echo "======================================================================"
        echo " 🎯 EVALUATING FPR FOR MODEL: $MODEL_NAME"
        echo "======================================================================"
        
        python evaluate_fpr.py --model "$MODEL_NAME"
        
        EXIT_CODE=$?
        sync

        if [ $EXIT_CODE -ne 0 ]; then
            echo "⚠️ WARNING: Utility evaluation failed for $MODEL_NAME (Exit Code: $EXIT_CODE)."
            FPR_HAS_ERRORS=1
        fi

        sleep 15 # VRAM Cooldown
    done
    
    echo "======================================================================"
    echo "✅ All FPR model evaluations complete."
    generate_master_charts
fi


# ------------------------------------------------------------
# HARDWARE STATUS AFTER EACH RUN
# ------------------------------------------------------------
echo "---------------------------------------------------------------"
echo "🏁 Exhaustive Execution Finished at $(date)"
echo "📊 Final GPU State:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader
echo "---------------------------------------------------------------"

exit 0