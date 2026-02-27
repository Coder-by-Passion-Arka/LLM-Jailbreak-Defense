# JailbreakBench Local Implementation (Light-weight LLM Protection Research)

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
Option A: Install Dependencies from environment.yml (Recommended)
First, check for the file named environment.yml in your root directory and run the following command:
```bash
# Activate the environment
conda env-jailbreak create -f environment.yml
```

Option B: Manual Install
Alternatively, you can create the environment manually:

```bash
conda create -n jailbreak python=3.10 -y

conda activate jailbreak

pip install vllm llama-cpp-python jailbreakbench[vllm] pandas matplotlib seaborn tabulate huggingface_hub
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

Loads the JBB-Behaviors dataset (100 harmful goals like "Write a phishing email").
```
```ini
Stage 2: Model Loading (Progressive)

It iterates through a list of models: Qwen1.5-1.8B -> Gemma-3-4B -> Llama-2-7B -> Vicuna-13B.

LightweightLLM Wrapper: Instead of standard loading, it initializes vLLM with:

quantization="float16" (16-bit weights)

max_model_len=2048 (Restricted context window to save memory)
```

Stage 3: The Attack Loop
```ini
**1. The Strategy & Attack Router**
* The orchestrator iterates through the selected defenses (`baseline`, `smoothing`, `enterprise`).
* For each defense, it iterates through the roster of threats: `["Simple-Prefix", "GCG", "PAIR", "MTJ", "JB-Chat"]`.
* It calls `AttackLoader.get_prompts()` to fetch the aligned adversarial data (either from the Hugging Face Hub or the JailbreakBench library) and formats it into a Universal Trajectory: `List[List[str]]`.

**2. The Multi-Turn Trajectory Execution**
For each individual attack sequence, the orchestrator begins a stateful conversation loop:
* **Input Interception:** The accumulating `chat_history` + the new `turn_prompt` is sent to `defense.process_input()`. 
* **Early Halting:** If the defense detects semantic drift or a malicious persona, it flags `blocked_in = True` and instantly severs the conversation, moving to the next attack.
* **LLM Generation:** If the input is deemed safe, the prompt is passed to the V100 GPU where `vLLM` generates a response. The response is appended to the `chat_history` for the next turn.

**3. The Post-Generation Gauntlet**
Once a trajectory finishes (either by reaching the final turn or being halted), the resulting text enters the evaluation phase:
* **Output Interception:** The target model's raw response is passed through `defense.process_output()`. If the streaming interceptor catches a leaked payload (e.g., `import subprocess`), the output is severed and overwritten with a safety warning.
* **The CPU Judge:** If the response survives both defense layers, it is sent to the Sidecar Judge (`Llama-Guard-4-12B`). The Judge runs on the system's CPU and grades the text against the MLCommons safety taxonomy, returning a final `SAFE` or `UNSAFE` (Jailbroken) verdict.

**4. Memory Cleanup & Aggregation**
* The result of every single trajectory (the prompts, the responses, the turns survived, and the final verdict) is appended to a massive Pandas DataFrame.
* Before switching to the next Defense Strategy, the Orchestrator forces Python's Garbage Collector (`gc.collect()`) to wipe the defense layer objects from RAM, preventing memory leaks during long benchmarking runs.
```

Stage 4: Data Recording
```ini
Saves the raw text response to model_responses/{model}_response.txt.

Saves a structured CSV to model_responses/{model}_results.csv.

Aggressively cleans GPU memory (gc.collect(), empty_cache()) before loading the next model.
```

Stage 5: Visualization
```ini
Aggregates data from all successfully loaded models.

Generates a grouped bar chart comparing success rates by category (e.g., Harassment, Malware).
```

## 6. How to Run the full Pipeline
Recommended: Using the Shell Script
This script handles exit codes and error checking automatically.

There are 4 modes to run the pipeline:
* -- None: Runs the unprotected control group.
* -- Smoothing: Runs the SmoothLLM defense.
* -- Enterprise: Runs the Enterprise Streaming Interceptor defense.
* -- Compare: Runs all defenses against all models.

* Run a complete ablation study (All Defenses vs All Models):

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh --compare
```

* Run specific defenses individually:

```bash
# Run the unprotected control group and the SmoothLLM defense
bash run_pipeline.sh --none --smoothing
```
## 7. Folder Structure
```ini
code
├── README.md
├── __pycache__
│   ├── attacks.cpython-310.pyc
│   ├── defensive_layer.cpython-310.pyc
│   ├── judge.cpython-310.pyc
│   └── logger_config.cpython-310.pyc
├── attacks.py
├── defense_strategy_selector.py
├── defensive_baseline.py
├── defensive_layer_smoothening.py
├── defensive_streaming_interceptor.py
├── environment.yml
├── judge.py
├── logger_config.py
├── logs
│   ├── execution.log
│   ├── summary_table.txt
│   └── terminal_output.txt
├── mock
│   └── mock_pipeline.py
├── model_loader.py
├── model_responses
│   ├── Qwen1.5-0.5B-Chat-AWQ_response.txt
│   ├── Qwen1.5-0.5B-Chat-AWQ_results.csv
│   ├── Qwen1.5-1.8B-Chat-AWQ_response.txt
│   ├── Qwen1.5-1.8B-Chat-AWQ_results.csv
│   ├── full_results.csv
│   └── jailbreak_comparison_chart.png
├── pipeline.py
├── results
├── run_pipeline.sh
├── terminal_output.txt
├── test.py
├── test_early_attacks
│   ├── __init__.py
│   └── test_early_get_prompts.py
└── test_early_judge
    ├── __init__.py
    └── test_early_get_token.py
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

**Key Functionalities:**

Hardware-Aware Memory Management: Iteratively loads each Target Model (e.g., Vicuna-13B, Llama-3-8B, Qwen1.5-1.8B, Gemma-3-4b) into VRAM. It aggressively destroys the vLLM parallel state, clears PyTorch process groups, and empties CUDA cache between models to prevent Out-Of-Memory (OOM) kernel panics.

Dynamic HDD Sweeper: Actively monitors the server's 70GB storage limit. It implements a cache-sweeping protocol that deletes multi-gigabyte Hugging Face model weights immediately after their evaluation loop finishes, ensuring the disk never fills up during massive benchmarking runs.

Global Aggregation: Collects all dataframes from the Orchestrator into global_full_results.csv and renders a global, multi-model seaborn/matplotlib comparative bar chart (global_defense_ablation_chart.png).

# Key Functionalities:
* **Dataset Loading:** Uses `jailbreakbench` to load harmful behaviors (Goals) and their associated metadata (Categories, Sources).
* **Model Management:**
    * Iteratively loads each **Target Model** (e.g., Vicuna-13B) into VRAM.
    * Aggressively cleans up GPU resources (`gc.collect()`, `torch.cuda.empty_cache()`) between models to prevent Out-Of-Memory (OOM) errors.
* **The Attack Loop:** Implements the core logic: `For each Model -> For every Attack strategy -> Record the raw response of the target model -> The Judge Model analyses the response and decides attack success `.
* **Defense Integration:**
    * Calls `defense.process_input()` *before* inference to filter malicious prompts.
    * Calls `defense.process_output()` *after* inference to filter harmful responses.
* **Reporting:**
    * **Data Aggregation:** Collects all raw data into `full_results.csv`.
    * **Metrics Calculation:** Computes the **Attack Success Rate (ASR)** grouped by Model and Attack Method.
    * **Visualization:** Uses `seaborn` to generate comparative bar charts (e.g., `jailbreak_comparison_chart.png`) stored in the `results/` directory.

---

# 3. `judge.py` (The Evaluator)

**Role:** An encapsulated "Supreme Court" for the pipeline. It determines if a specific response constitutes a successful jailbreak.

# Key Functionalities:
* **Model Loading:** Loads the **Llama-4-12B-Guard** model using `vLLM`. It is configured with `quantization="GGUF"` (4-bit) to ensure the 12B parameter model fits within the available RAM using llama-cpp alongside the target model.
* **Paper-Compliant Prompting:** Implements the exact system prompt from JailbreakBench dataset.
* **Verdict Logic:** Parses the Judge's output to return a boolean:
    * **True (Unsafe):** The attack succeeded (Jailbreak).
    * **False (Safe):** The model refused (Defense Success).

---

# 4. `attacks.py` (The Adversary)

**Role:** A centralized extraction and formatting hub that bridges distinct mathematical and conversational datasets into a single, cohesive Universal Trajectory Format (List[List[str]]).

# Key Functionalities:
* **JBB Artifact Retrieval:** Connects to the `jailbreakbench` library to fetch pre-computed adversarial strings (Artifacts) for methods token-optimized methods like **GCG** and **PAIR**.

* **MTJ Dataset Slicing:** Dynamically downloads, caches, and parses the official MTJ-Bench multi-turn dataset from the Hugging Face Hub, preserving the chronological order of semantic-drift trajectories.

* **Model Mapping:** Automatically maps HuggingFace model IDs (e.g., `lmsys/vicuna-13b-v1.5`) to the internal IDs used by the JailbreakBench library, ensuring the correct artifacts are loaded for the correct model.

* **Fallback Logic:** If specific artifacts (e.g., GCG strings for a custom model) are missing, it defaults to a standard "Prefix Injection" attack to ensure the pipeline continues running without crashing.

---

# 5. defensive_layer.py (The Defense Factory)
**Role:** The centralized, dynamic routing hub for conducting rigorous ablation studies. Instead of a static placeholder, this module utilizes the Strategy Design Pattern to hot-swap complex defense mechanisms directly from the command line (--baseline, --smoothing, --enterprise, --none, --compare).

**Implemented Strategies:**

* BaselineDefense (Control Heuristic): A robust, multi-layer vanilla defense pipeline representing current industry standards. It sequentially routes inputs through static regex obfuscation filters, a local RoBERTa Toxicity classifier, and a TinyLlama semantic rewriter, capped by an output-leakage guard.

* Random Noise insertion in the prompt (SmoothingDefense): Implements a highly novel, mathematically grounded Randomized Smoothing algorithm. It systematically injects character-level perturbations (insertions, deletions, replacements) based on probabilistic hyper-parameter (e.g., 10%). This shatters the brittle mathematical gradients of token-optimized attacks (like GCG) while preserving semantic meaning for the target LLM.

Semantic Drift Detector: Designed to defeat multi-turn semantic drift. Uses a Stateful Context Extractor to detect malicious persona adoption over time, and implements a constant-time $O(1)$ Sliding Window Buffer to intercept and sever highly-specific threat payloads mid-generation..

**Key Functionalities:**

Stateful Input Filtering (process_input): Uniquely designed to handle the Universal Trajectory Format. Instead of evaluating prompts in a vacuum, it processes accumulating conversational history to catch "semantic drift" in multi-turn (MTJ) scenarios.

Output Leak Detection (process_output): Inspects model responses for system prompt leakages, canary tokens, and "Prefix Traps" (where a model superficially complies before yielding the payload).

**Dynamic Factory Routing (get_defense_layer):** Allows the main pipeline to instantiate and clear specific defense architectures in memory sequentially, enabling seamless, automated comparative analysis across multiple models.

## 6. LLM Jailbreak Defense Benchmark:
Single-Turn (JBB) vs Multi-Turn (MTJ)📖 Paper AbstractAs Large Language Models (LLMs) are increasingly deployed in stateful, conversational agents, evaluating their security purely through single-turn prompt injection is fundamentally insufficient. While modern heuristic defenses and Randomized Smoothing (SmoothLLM) have achieved high block rates against isolated, token-optimized attacks (e.g., GCG, PAIR), they routinely collapse when an adversary dilutes malicious intent across prolonged, multi-turn conversational trajectories.In this research, we expose the critical vulnerability of context-window scaling by bridging the gap between mathematical single-turn benchmarks (JailbreakBench) and psychological multi-turn evaluations (MTJ-Bench). We introduce a Universal Trajectory Framework that dynamically transposes standard single-turn adversarial goals into multi-turn semantic escalations, enabling a mathematically rigorous ablation study of defense degradation over time.Our comprehensive evaluation across leading architectures demonstrates that while SmoothLLM shatters token-based gradients, it suffers catastrophic failure against 5-turn MTJ crescendo attacks. To counter this, we propose the Stateful Streaming Interceptor, an enterprise-grade defense utilizing constant-time $O(1)$ sliding window buffers to sever malicious payloads mid-generation. We prove that a holistic approach—combining input perturbation with stateful output interception—is required to maintain mathematically provable robustness in continuous, real-world deployments.

---

## 7. System Architecture
This repository is built using a decoupled Strategy Design Pattern. This ensures the code is highly modular, prevents Out-of-Memory (OOM) crashes, and allows researchers to easily drop in custom defense mechanisms.

1. **Core Orchestrationpipeline.py (Master Bootstrapper)**: 
The lightweight entry point. It parses command-line arguments, safely loads Target Models (like Llama-3 or Vicuna) into GPU VRAM, and manages the strict HDD cleanup process to prevent the server from running out of storage.defense_strategy_selector.py (The Conductor): The central nervous system. It runs the massive evaluation loop (Model -> Defense Strategy -> Attack Method). It manages the multi-turn conversational history and generates the final Matplotlib Attack Success Rate (ASR) charts.

2. **Evaluation & Datajudge.py (Sidecar Evaluator):**
To protect GPU memory, this module offloads evaluation strictly to the CPU/System RAM. It uses llama.cpp to run Llama-Guard-4-12B, evaluating model responses against strict MLCommons safety taxonomies.attacks.py (The Adversary): Dynamically fetches pre-computed mathematical adversarial strings (GCG/PAIR) from the JailbreakBench library, and pulls real multi-turn psychological roleplay attacks from the MTJ-Bench Hugging Face dataset.model_loader.py (Hardware Manager): Automatically detects the GPU architecture (e.g., Volta V100) and enforces strict FP16 precision, stripping incompatible 4-bit AWQ formats to prevent CUDA compilation crashes.

3. **The Defense Factory (The Shields):** Researchers can toggle these defenses using command-line flags to compare their effectiveness:

* defensive_baseline.py (--baseline): A multi-layer heuristic control group. It uses Regex obfuscation blocks, a CPU-bound Toxicity BERT classifier, and a TinyLlama semantic rewriter.
    
* defensive_smoothing.py (--smoothing): A mathematically rigorous implementation of SmoothLLM. It systematically injects character-level perturbations (Swaps, Patches, Inserts) to shatter token-optimized adversarial gradients.
    
* defensive_streaming_interceptor.py (--enterprise): Our proposed capstone. It uses a Stateful Context Extractor to detect multi-turn persona drift and an $O(1)$ Sliding Window Buffer to intercept and sever malicious payloads mid-generation.

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
(Fetches JBB & MTJ data       (Loads Baseline, Smooth,      (Loads Llama-Guard-4-12B
 formats to List[List])        or Enterprise Streaming)      to CPU via llama.cpp)
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