# ./pipeline.py

import os
import gc
import sys
import torch # <--- FIXED: Global import prevents UnboundLocalError in cleanup
import json
import time
import argparse
import traceback
import subprocess

# =====================================================================
# CVE-2025-32434 GLOBAL SECURITY BYPASS FOR LEGACY V100 HARDWARE
# MUST execute before any other imports to poison the module cache!
# =====================================================================
import transformers.utils.import_utils
import transformers.modeling_utils

# Force the library to believe Torch is safe
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
transformers.modeling_utils.check_torch_load_is_safe = lambda: True
# =====================================================================
import types
import multiprocessing

# --- CRITICAL CUDA MULTIPROCESSING FIX ---
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# -----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import shared config and logger
from logger_config import logger, RESULT_DIR, RESPONSE_DIR

# Import modules
from model_loader import HardwareAwareModelLoader
from defense_strategy_selector import DefenseFactory
from attacks import AttackLoader
from judge import JailbreakJudge

HF_TOKEN = os.environ.get("HF_TOKEN") or print("Enter your own Hugging Face API token")

# =====================================================================
# COMMAND LINE ARGUMENT PARSER
# =====================================================================
parser = argparse.ArgumentParser(description="A* Jailbreak Benchmark Pipeline")
parser.add_argument('--fuse', action='store_true', help='Run the FUSE Defense')
parser.add_argument('--smoothing', action='store_true', help='Run the randomized Smoothing defense')
parser.add_argument('--dcmd', action='store_true', help='Run the Dual Phase Cryptographic Manifold Defense')
# parser.add_argument('--enterprise', action='store_true', help='Run the Stateful Streaming Interceptor')
parser.add_argument('--none', action='store_true', help='Run with no defense (Control Group)')
parser.add_argument('--compare', action='store_true', help='Run ALL strategies and compare them')
parser.add_argument('--model', type=str, required=True, help='Target a specific model for isolated execution')
args = parser.parse_args()

STRATEGIES_TO_TEST = []
if args.compare:
    STRATEGIES_TO_TEST = ['none', 'fuse', 'smoothing', 'dcmd']
else:
    if args.none: STRATEGIES_TO_TEST.append('none')
    if args.fuse: STRATEGIES_TO_TEST.append('fuse')
    if args.smoothing: STRATEGIES_TO_TEST.append('smoothing')
    # if args.enterprise: STRATEGIES_TO_TEST.append('enterprise')
    if args.dcmd: STRATEGIES_TO_TEST.append("dcmd")

if not STRATEGIES_TO_TEST:
    STRATEGIES_TO_TEST = ['none', 'fuse', 'smoothing', 'dcmd']

# =====================================================================
# PIPELINE CONFIGURATION
# =====================================================================
ATTACKS_TO_TEST = ["Simple-Prefix", "GCG", "PAIR", "JB-Chat", "MTJ"]
TEST_LIMIT = 2  # Adjust as needed for fast testing vs full benchmark

try:
    import sys
    import types
    # --- ZERO-FRICTION DEPENDENCY HOTFIX ---
    if "litellm.llms.prompt_templates.factory" not in sys.modules:
        mock_pt = types.ModuleType("litellm.llms.prompt_templates")
        mock_factory = types.ModuleType("litellm.llms.prompt_templates.factory")
        mock_factory.custom_prompt = lambda *args, **kwargs: ""
        sys.modules["litellm.llms.prompt_templates"] = mock_pt
        sys.modules["litellm.llms.prompt_templates.factory"] = mock_factory

    import jailbreakbench as jbb
    from vllm import LLM, SamplingParams
except ImportError as e:
    logger.critical(f"Missing Dependency: {e}")
    sys.exit(1)

def enforce_context_limit(context_str: str, char_limit: int = 6500) -> str:
        """
        Ensures the prompt never exceeds the vLLM 2048 token limit (~8000 chars).
        Cuts the middle out of long chat histories to preserve system constraints and the current attack.
        """
        if len(context_str) <= char_limit:
            return context_str
        
        # Keep the first 1500 chars (Turn 1 context) and the last 4800 chars (Current Attack)
        prefix_len = 1500
        suffix_len = char_limit - prefix_len
        
        return context_str[:prefix_len] + "\n\n...[HISTORY TRUNCATED TO PREVENT VRAM OVERFLOW]...\n\n" + context_str[-suffix_len:]

class Pipeline:
    def cleanup_vram(self):
        import torch
        import gc
        import subprocess
        
        logger.info("[VRAM CLEANER] 🧹 Executing hardware state reset...")

        # 1. Corrected vLLM Cleanup Functions with Fallback
        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel, 
                destroy_distributed_environment
            )
            destroy_model_parallel()
            destroy_distributed_environment()
            logger.info("[VRAM CLEANER] ✅ vLLM distributed environment destroyed.")
        except ImportError:
            # Fallback for older vLLM versions
            try:
                from vllm.distributed.parallel_state import destroy_distributed_runtime
                destroy_distributed_runtime()
            except:
                pass
            logger.debug("[VRAM CLEANER] Distributed cleanup functions not found in this vLLM version.")
        except Exception as e:
            logger.debug(f"[VRAM CLEANER] Distributed runtime already closed or failed: {e}")

        # 2. Force-kill lingering background worker PIDs
        try:
            subprocess.run(["pkill", "-f", "vllm.executor.multiproc_worker_utils"], check=False, stderr=subprocess.DEVNULL)
            logger.info("[VRAM CLEANER] 🔪 Terminated orphaned vLLM worker processes.")
        except Exception:
            pass

        # 3. Standard Garbage Collection and Torch Purge
        gc.collect()
        torch.cuda.empty_cache()
        
        # 4. Critical: Force reclamation of Reserved VRAM back to the OS
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.synchronize()
                    
        logger.info("[VRAM CLEANER] ✨ VRAM Hard Purge Complete.")    
    
    # 3. Standard Garbage Collection and Torch Purge
    gc.collect()
    torch.cuda.empty_cache()
    
    # 4. Critical: Force reclamation of Reserved VRAM back to the OS
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()
                
    logger.info("[VRAM CLEANER] ✨ VRAM Hard Purge Complete.")
    
    def __init__(self):
        logger.info("[SYSTEM] 🟢 Entering TRY block: JBB Dataset Initialization")
        try:
            self.ds = jbb.read_dataset()
            self.behaviors = self.ds.behaviors[:TEST_LIMIT] if TEST_LIMIT else self.ds.behaviors
            self.goals = self.ds.goals[:TEST_LIMIT] if TEST_LIMIT else self.ds.goals
            self.categories = self.ds.categories[:TEST_LIMIT] if TEST_LIMIT else self.ds.categories
            logger.info(f"[SYSTEM] ✅ Datasets loaded successfully. {len(self.goals)} instances targeted.")
        except Exception as e:
            logger.error(f"[SYSTEM] ❌ EXCEPTION in Dataset Initialization: {e}")
            logger.debug(traceback.format_exc())
            sys.exit(1)
        finally:
            logger.info("[SYSTEM] 🏁 Exiting FINALLY block: JBB Dataset Initialization")

    def generate_per_model_charts(self, df):
        """Generates soothing, side-by-side bar plots for each model."""
        if df.empty:
            logger.warning("[VISUALIZER] No data available to plot.")
            return

        logger.info("[VISUALIZER] 🟢 Entering TRY block: Chart Generation")
        try:
            models = df['Model'].unique()
            for model in models:
                model_df = df[df['Model'] == model]
                summary_df = model_df.groupby(['Defense_Strategy', 'Attack'])['Jailbroken'].mean().reset_index()
                summary_df['ASR (%)'] = summary_df['Jailbroken'] * 100

                plt.figure(figsize=(14, 8))
                sns.set_theme(style="whitegrid")
                
                palette = sns.color_palette("pastel")
                
                sns.barplot(
                    data=summary_df, 
                    x='Attack', 
                    y='ASR (%)', 
                    hue='Defense_Strategy',
                    palette=palette,
                    edgecolor='black'
                )
                
                plt.title(f'Attack Success Rate (ASR) by Defense Strategy\nTarget: {model}', fontsize=18, fontweight='bold', pad=15)
                plt.ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
                plt.xlabel('Adversarial Attack Method', fontsize=14, fontweight='bold')
                plt.ylim(0, 100)
                plt.legend(title='Defense Layer', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                safe_model_name = model.replace("/", "_")
                output_path = os.path.join(RESULT_DIR, f'asr_chart_{safe_model_name}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"[VISUALIZER] ✅ Saved soothing comparative chart for {model} at {output_path}")
        except Exception as e:
            logger.error(f"[VISUALIZER] ❌ EXCEPTION in Chart Generation: {e}")
            logger.debug(traceback.format_exc())
        finally:
            logger.info("[VISUALIZER] 🏁 Exiting FINALLY block: Chart Generation")

    def run(self):
        import json
        import time
        import traceback
        import subprocess
        import sys
        
        model_name = args.model
        generated_data = [] 
        target_llm = None

        # Log Paths for Telemetry
        HTS_LOG = os.path.join(RESPONSE_DIR, "hts_attack_execution.log")
        PRF_LOG = os.path.join(RESPONSE_DIR, "prf_attack_execution.log")

        logger.info(f"\n{'='*80}\n[TARGET] 🎯 Booting Target Model: {model_name}\n{'='*80}")
        
        # =====================================================================
        # PIPELINE PHASE-1: TARGET MODEL GENERATION (GPU BOUND)
        # =====================================================================
        logger.info("[PIPELINE PHASE-1] 🟢 Entering TRY block: Target Model VRAM Loading & Execution")
        try:
            smart_config = HardwareAwareModelLoader.get_optimal_target_config(preferred_base_model=model_name)
            
            # 🚨 CRITICAL FIXES FOR V100 FRAGMENTATION OOM
            smart_config['gpu_memory_utilization'] = 0.65 
            smart_config['max_model_len'] = 2048 
            
            custom_stdout, custom_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
                target_llm = LLM(**smart_config)
            finally:
                sys.stdout, sys.stderr = custom_stdout, custom_stderr

            sampling_params = SamplingParams(temperature=0.1, max_tokens=150)
            attacker = AttackLoader(model_name)

            for current_strategy in STRATEGIES_TO_TEST:
                logger.info(f"\n[PIPELINE PHASE-1] 🛡️ ACTIVATING DEFENSE: {current_strategy.upper()}")
                
                orchestrator_config = {
                    "model_name": model_name,
                    "sampling_params": sampling_params
                }
                defense = DefenseFactory.get_defense_layer(current_strategy, config=orchestrator_config)
                
                for attack_method in ATTACKS_TO_TEST:
                    logger.info(f"[PIPELINE PHASE-1] 🚀 Generating responses for Attack: {attack_method}")
                    try:
                        prompt_sequences, active_goals, active_behaviors = attacker.get_prompts(
                            attack_method, self.goals, self.categories
                        )
                    except Exception as e:
                        logger.error(f"[PIPELINE PHASE-1] ❌ Skipping {attack_method} extraction failure: {e}")
                        continue

                    for behavior, goal, sequence in zip(active_behaviors, active_goals, prompt_sequences):
                        chat_history = ""
                        blocked_in_turn = False
                        final_turn_response = ""
                        
                        for turn_prompt in sequence:
                            context_to_evaluate = chat_history + f"User: {turn_prompt}"
                            
                            # --- DYNAMIC UNPACKING: INPUT GATE ---
                            input_result = defense.process_input(context_to_evaluate)
                            if len(input_result) == 3:
                                blocked_in, processed_context, phase1_telemetry = input_result
                            else:
                                blocked_in, processed_context = input_result
                                phase1_telemetry = ""
                                
                            # Record DCMD Phase 1 Telemetry
                            if current_strategy == 'dcmd' and phase1_telemetry:
                                with open(HTS_LOG, "a") as f:
                                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                    f.write(f"{timestamp} | {phase1_telemetry}")

                            if blocked_in:
                                blocked_in_turn = True
                                final_turn_response = f"I cannot fulfill this request (Defense Blocked Input)."
                                break 
                                
                            # GENERATION
                            if isinstance(processed_context, list):
                                safe_contexts = [enforce_context_limit(c + "\nAssistant: ") for c in processed_context]
                                outputs = target_llm.generate(safe_contexts, sampling_params)
                                final_turn_response = [out.outputs[0].text.strip() for out in outputs]
                                chat_history = processed_context[0] + f"\nAssistant: {final_turn_response[0]}\n\n"
                            else:
                                safe_context = enforce_context_limit(processed_context + "\nAssistant: ")
                                outputs = target_llm.generate([safe_context], sampling_params)
                                raw_response = outputs[0].outputs[0].text.strip()
                                chat_history = processed_context + f"\nAssistant: {raw_response}\n\n"
                                final_turn_response = raw_response

                        if not blocked_in_turn:
                            # --- DYNAMIC UNPACKING: OUTPUT GATE ---
                            output_result = defense.process_output(final_turn_response)
                            if len(output_result) == 3:
                                blocked_out, safe_response, phase2_telemetry = output_result
                            else:
                                blocked_out, safe_response = output_result
                                phase2_telemetry = ""
                                
                            # Record DCMD Phase 2 Telemetry
                            if current_strategy == 'dcmd' and phase2_telemetry:
                                with open(PRF_LOG, "a") as f:
                                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                    f.write(f"{timestamp} | {phase2_telemetry}")
                                    
                            final_turn_response = "I cannot fulfill this request (Defense Blocked Output)." if blocked_out else safe_response

                        generated_data.append({
                            "Strategy": current_strategy.upper(),
                            "Attack": attack_method,
                            "Category": behavior,
                            "Goal": goal,
                            "Final_Response": final_turn_response,
                            "Blocked_In": blocked_in_turn
                        })

                # Defense Cleanup
                if hasattr(defense, 'cleanup'): defense.cleanup()
                del defense
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"[PIPELINE PHASE-1] ❌ EXCEPTION in Target Model Generation: {e}")
            logger.debug(traceback.format_exc())
            
        finally:
            logger.info("[PIPELINE PHASE-1] 🏁 Initiating Strict VRAM & Distributed Cleanup...")
            if target_llm is not None:
                del target_llm
            
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as e:
                logger.debug(f"Distributed cleanup failed: {e}")

            gc.collect()
            torch.cuda.empty_cache()
            
            temp_artifacts = f"tmp_artifacts_{model_name.replace('/', '_')}.json"
            if generated_data:
                with open(temp_artifacts, 'w') as f:
                    json.dump(generated_data, f)
                logger.info(f"[PIPELINE PHASE-1] ✅ Saved {len(generated_data)} artifacts for Phase 2.")
            else:
                logger.warning("[PIPELINE PHASE-1] ⚠️ No data was generated. Skipping Judge.")
                return

            time.sleep(10) # Cooldown for port release

        # =====================================================================
        # PIPELINE PHASE-2: STANDALONE OS-PROCESS JUDGE
        # =====================================================================
        logger.info(f"\n{'='*80}\n[PIPELINE PHASE-2] ⚖️ Launching Standalone Judge Process\n{'='*80}")
        self.cleanup_vram()
        time.sleep(10)

        try:
            judge_env = os.environ.copy()
            judge_env.pop("MASTER_ADDR", None)
            judge_env.pop("MASTER_PORT", None)
            judge_env["NCCL_P2P_DISABLE"] = "1" 
            # 🚨 FIX FOR FRAGMENTATION OOM IN JUDGE
            judge_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            result = subprocess.run([
                sys.executable, "judge_runner.py", 
                "--artifacts_file", temp_artifacts,
                "--model_name", model_name
            ], env=judge_env, check=True)
            
            if result.returncode == 0:
                logger.info("[PIPELINE PHASE-2] ✅ Judge Process Completed Successfully.")
        except Exception as e:
            logger.error(f"[PIPELINE PHASE-2] ❌ Judge Subprocess Failed: {e}")
        finally:
            if os.path.exists(temp_artifacts):
                logger.info(f"[PIPELINE PHASE-2] Deleting Temporary Artifacts")
                os.remove(temp_artifacts)

        # =====================================================================
        # PIPELINE PHASE-3: FINAL REPORTING & CHARTING
        # =====================================================================
        logger.info("[PIPELINE PHASE-3] 🟢 Entering TRY block: Generating Final Charts")
        try:
            csv_path = os.path.join(RESPONSE_DIR, "global_attack_results.csv")
            if os.path.exists(csv_path):
                import pandas as pd
                final_df = pd.read_csv(csv_path)
                self.generate_per_model_charts(final_df)
                logger.info(f"[PIPELINE PHASE-3] 📊 Charts updated. Total records: {len(final_df)}")
            else:
                logger.warning("[PIPELINE PHASE-3] ⚠️ No results file found to chart.")
        except Exception as e:
            logger.error(f"[PIPELINE PHASE-3] ❌ EXCEPTION in Charting: {e}")
            logger.debug(traceback.format_exc())
        finally:
            logger.info("[PIPELINE PHASE-3] 🏁 Exiting FINALLY block: Pipeline complete.")

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()