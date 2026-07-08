# JailbreakBench Local Implementation (Low-VRAM Optimized)

This repository contains a robust, local implementation of the [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) framework. It has been heavily modified to run on consumer hardware (specifically 4GB+ VRAM GPUs) using **WSL2** and **vLLM** with aggressive quantization.

## Features Implemented
* **Progressive Loading:** Attempts to load models from smallest (0.5B) to largest (7B+).
* **Low-Memory Optimization:** Uses 4-bit AWQ quantization, 2k context windows, and aggressive GPU cache clearing to prevent OOM crashes.
* **Dual Logging:** Streams output to both the terminal and a structured log file (`terminal_output.txt`).
* **Structured Data:** Saves attack results to individual CSVs for analysis.
* **Auto-Visualization:** Automatically generates comparative bar charts (`jailbreak_comparison_chart.png`) at the end of the run.

## 1. Prerequisites (Windows Users)
You must be running **WSL2** (Ubuntu 20.04 or 22.04 recommended) to use `vLLM`, as it is not natively supported on Windows.
* Ensure you have the latest **NVIDIA Drivers** installed on Windows.
* Run `wsl --install` in PowerShell if you haven't set up WSL yet.

## 2. Install Conda
If you do not have Conda installed in your WSL environment, run the following commands:

```bash
# 1. Update Ubuntu packages
sudo apt update && sudo apt upgrade -y

# 2. Download Miniconda installer
wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

# 3. Install it (Follow the prompts, type 'yes' when asked)
bash Miniconda3-latest-Linux-x86_64.sh

# 4. Refresh your shell to make 'conda' command available
source ~/.bashrc
```

## 3. Environment Setup
Option A: Install Dependencies from env.yml (Recommended)
First, check for the file named env.yml in your root directory and run the following command:
```bash
# Activate the environment
conda env-jailbreak create -f env.yml

```

Option B: Manual Install
Alternatively, you can create the environment manually:

```bash
conda create -n env-jailbreak python=3.10 -y
conda activate env-jailbreak
pip install "jailbreakbench[vllm]"
```

## 4. Verification
Run these commands to ensure your GPU is detected and libraries are installed correctly.

```bash
# 1. Check if NVIDIA driver is accessible from WSL
nvidia-smi

# 2. Check if PyTorch can see the GPU
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# 3. Check if vLLM is installed successfully
python3 -c "import vllm; print('vLLM imported successfully')"
```

## 5. Pipeline Workflow (How it Works)
The main.py script orchestrates a 5-stage pipeline:

Stage 1: Initialization
```ini
Sets up a DualLogger to capture all print statements to terminal_output.txt.

Loads the attacks from various datasets like
 * JBB-Behaviors dataset (100 harmful goals like "Write a phishing email").
 * AdvBench Harmful Behaviours
 * Alpaca Benign Prompts
```
```ini
Stage 2: Model Loading (Sequential)

It iterates through a list of models: Qwen-1.5B-Instruct -> Gemma-2b-it -> Llama-2-7B -> Vicuna-13B-v1.5.

LightweightLLM Wrapper: Instead of standard model loading, our framework utilizes vLLM with the optimised versions of the above models which ensures quicker loading and faster inference time:

quantization=__dynamic_quant_selection_strategy (Look into 'model_loader.py') for more info

max_model_len=4096 (Restricted context window to save memory)

enforce_eager=True (Disables CUDA graphs to save ~500MB VRAM)
```

Stage 3: The Attack Loop
```ini

A dataset is prepared for each model to test. The dataset is auto-generated from the standard datasets mentioned above (For details look into 'dataset_builder.py')

Framework queries the model and records the response.

Expert Judge: Llama-Guard-3-8B model is used to check if the response contains refusal keywords like ("I cannot", "illegal"). If not, it counts as a 'SUCCESSFUL JAILBREAK' which means that the defensive layer has failed to protect the LLM.
```

Stage 4: Data Recording
```ini
Saves the raw text response to model_responses/{model}_response.txt.

Saves a structured CSV to model_responses/{model}_results.csv.

Aggressively cleans GPU memory (gc.collect(), empty_cache()) before loading the next model.
```
Stage 5: Visualization
```ini
Aggregates data from all successfully loaded models. (Look into 'visualizer.py')

Generates a grouped heatmaps showcasing 'Attack Success Rates' &  'False Positive Rates'.
```

## 6. How to Run the full Pipeline
Recommended: Using the Shell Script
This script handles exit codes and error checking automatically.

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Alternative: Direct Python

```bash
python pipeline.py
```

## 7. Folder Structure
```ini
root
├── LICENSE
├── code
│   ├── README.md
│   ├── attacks.py
│   ├── dataset_builder.py
│   ├── defense_strategy_selector.py
│   ├── defensive_fuse.py
│   ├── defensive_dual_phase_cryptographic_manifold_defense.py
│   ├── defensive_layer.py
│   ├── defensive_smoothing.py
│   ├── defensive_streaming_interceptor.py
│   ├── env.yml
│   ├── evaluate_fpr.py
│   ├── judge.py
│   ├── judge_runner.py
│   ├── logger_config.py
│   ├── logit_watermarking.py
│   ├── model_loader.py
│   ├── orchastrate.py
│   ├── pipeline.py
│   ├── run_pipeline.sh
│   ├── train_hts.py
│   └── visualizer.py
```
## 8. Purpose of each Module in the Pipeline

# 1. `logger_config.py` (Configuration & Logging)

**Role:** The centralized configuration hub for the entire pipeline. It ensures consistent logging and directory management across all other modules.

# Key Functionalities:
* **Directory Setup:** Automatically checks for and creates the following directory structure upon execution:
    * `./logs/`: Stores terminal output and execution logs.
    * `./results/`: Stores final metrics, summary tables, and visualization plots.
    * `./model_responses/`: Stores raw CSV data containing every prompt, response, and verdict.
* **DualLogger Class:** Implements a custom `sys.stdout` wrapper that splits all console output into two streams:
    1.  **Terminal:** Visible real-time to the user.
    2.  **File (`terminal_output.txt`):** Saved permanently for debugging.
* **Global Logger Instance:** Initializes a Python `logging` instance ("JailbreakPipeline") that is imported by all other modules (`judge.py`, `attacks.py`, etc.). This prevents log fragmentation and ensures all events are recorded in `execution.log`.

---

# 2. `pipeline.py` (Master Orchestrator)

**Role:** The central controller that manages the workflow. It connects the Dataset, Target Model, Attacker, Defense Layer, and Judge.

# Key Functionalities:
* **Dataset Loading:** Uses `jailbreakbench` to load harmful behaviors (Goals) and their associated metadata (Categories, Sources).
* **Model Management:**
    * Iteratively loads each **Target Model** (e.g., Vicuna-13B) into VRAM.
    * Aggressively cleans up GPU resources (`gc.collect()`, `torch.cuda.empty_cache()`) between models to prevent Out-Of-Memory (OOM) errors.
* **The Attack Loop:** Implements the core logic: 
    ```
    for each Model Selection -> 
        for each Attack Selection -> 
            (Attack passed through Defensive Layer)
            Model Inference
            Response Stored
        for each response Performance Judgement -> 
            Performance results Stored
    Performance heatmap 
    ```.
* **Defense Integration:**
    * Calls `defense.process_input()` *before* inference to filter malicious prompts.
    * Calls `defense.process_output()` *after* inference to filter harmful responses.
* **Reporting:**
    * **Data Aggregation:** Collects all raw data into `full_response.csv`.
    * **Metrics Calculation:** Computes the **Attack Success Rate (ASR)** grouped by Model and Attack Method.
    * **Visualization:** Uses `seaborn` to generate comparative bar charts (e.g., `jailbreak_comparison_chart.png`) stored in the `results/` directory.

---

# 3. `judge.py` (The Evaluator)

**Role:** An encapsulated "Supreme Court" for the pipeline. It determines if a specific response constitutes a successful jailbreak.

# Key Functionalities:
* **Model Loading:** Loads the **Llama-3-70B-Instruct** model using `vLLM`. It is configured with `quantization="awq"` (4-bit) to ensure the 70B parameter model fits within the available VRAM alongside the target model.
* **Paper-Compliant Prompting:** Implements the exact system prompt from **Table 10** of the JailbreakBench paper. This ensures the evaluation criteria (e.g., "The RESPONSE must clearly provide specific details...") matches the official benchmark.
* **Verdict Logic:** Parses the Judge's output to return a boolean:
    * **True (Unsafe):** The attack succeeded (Jailbreak).
    * **False (Safe):** The model refused (Defense Success).

---

# 4. `attacks.py` (The Adversary)

**Role:** Abstracts the complexity of generating or retrieving adversarial prompts.

# Key Functionalities:
* **Artifact Retrieval:** Connects to the `jailbreakbench` library to fetch pre-computed adversarial strings (Artifacts) for methods like **GCG** and **PAIR**.
* **Model Mapping:** Automatically maps HuggingFace model IDs (e.g., `lmsys/vicuna-13b-v1.5`) to the internal IDs used by the JailbreakBench library, ensuring the correct artifacts are loaded for the correct model.
* **Fallback Logic:** If specific artifacts (e.g., GCG strings for a custom model) are missing, it defaults to a standard "Prefix Injection" attack to ensure the pipeline continues running without crashing.

---

# 5. `defensive_layer.py` (The Shield)

**Role:** A placeholder module designed for future implementation of custom defense mechanisms.

# Key Functionalities:
* **Input Filtering (`process_input`):** A hook to inspect or modify the user's prompt *before* it reaches the target model. Currently returns `False` (allow) by default.
* **Output Filtering (`process_output`):** A hook to inspect or modify the model's response *before* it is shown to the user or judge. Currently returns `False` (allow) by default.
* **Extensibility:** This module is decoupled from the main pipeline, allowing researchers to implement complex defenses (e.g., Perplexity Filtering, SmoothLLM) without modifying the core orchestration logic.

## 9. Flowchart of the Pipeline
```ini
+-----------------------------------------------------------------------------------+
|                                 PIPELINE.PY                                       |
|                 (Bootstrapper, CLI, & Hardware Memory Manager)                    |
+---------------------------------------+-------------------------------------------+
                                        |
  [STEP 1: H/W CONFIG & LOAD]           |  (Uses model_loader.py for V100 FP16 Check)
                                        v
+-----------------------------------------------------------------------------------+
|                         DEFENSE_STRATEGY_SELECTOR.PY                              |
|                      (The Core Orchestrator & Factory Hub)                        |
+---------------------------------------+-------------------------------------------+
                                        |                              |
  [STEP 2: GATHER RESOURCES]            |                              |
         v                              v                              v
+----------------------+      +----------------------+      +----------------------+
|      ATTACKS.PY      |      |   DEFENSE FACTORY    |      |       JUDGE.PY       |
| (Adversary Loader)   |      | (Strategy Pattern)   |      | (Sidecar Evaluator)  |
+----------------------+      +----------------------+      +----------------------+
         |                              |                              |
         v                              v                              v
(Fetches JBB & MTJ data       (Loads Baseline, Smooth,      (Loads Llama-Guard-3-8B
 & Benign Alpaca data)        or Enterprise Streaming)        to CPU via llama.cpp)
         |                              |                              |
         +----------------+             |             +----------------+
                          |             |             |
                          v             v             v
+-----------------------------------------------------------------------------------+
|                          STATEFUL EXECUTION LOOP                                  |
+-----------------------------------------------------------------------------------+
| 1. Orchestrator -> Iterates through: Defenses -> Attacks -> Trajectories          |
| 2. Orchestrator -> Fetches Multi-Turn sequence (List[str]) from attacks.py        |
|                                                                                   |
| +--[ TURN LOOP BEGINS ]---------------------------------------------------------+ |
| | 3. Orchestrator -> Sends accumulating Chat History to Defense (process_input) | |
| | 4. Defense      -> Scans full history. If safe, returns isolated latest turn. | |
| | 5. Orchestrator -> Sends safe turn to TARGET MODEL (vLLM on V100 GPU)         | |
| | 6. Orchestrator <- Receives Raw Response. Appends to Chat History.            | |
| +--[ TURN LOOP ENDS ]-----------------------------------------------------------+ |
|                                                                                   |
| 7. Orchestrator -> Sends Final Output to Defense (process_output / interceptor)   |
| 8. Orchestrator -> Sends survived response + original Goal to judge.py            |
| 9. judge.py     -> Inference on Llama-Guard-4 (Strictly on CPU / System RAM)      |
| 10.Orchestrator <- Receives Verdict (Boolean: Jailbroken or Safe)                 |
+-----------------------------------------------------------------------------------+
                                        |
  [STEP 3: CLEANUP & REPORT]            |  (pipeline.py clears HDD & GPU VRAM)
                                        v
+----------------------+      +----------------------+      +----------------------+
| global_results.csv   |      |   ASR Matrix PNGs    |      | Global Ablation PNG  |
|  (Raw Data Logging)  |      |  (Per-Model Stats)   |      |  (Cross-Model Bar)   |
+----------------------+      +----------------------+      +----------------------+
```