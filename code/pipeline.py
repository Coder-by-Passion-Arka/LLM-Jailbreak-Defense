# Version - 3
# # ./pipeline.py
# import os
# import sys
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# import gc
# import time
# from tabulate import tabulate # Optional: for pretty printing, but we can use pandas markdown too

# # Import shared config and logger
# from logger_config import logger, LOGS_DIR, RESULT_DIR, OUTPUT_DIR

# # Import modules
# from judge import JailbreakJudge
# from attacks import AttackLoader
# # from defensive_layer_vanilla import DefenseLayer
# # from defensive_layer_smoothening import DefenseLayer
# from model_loader import HardwareAwareModelLoader
# import argparse
# from defensive_layer import get_defense_layer # Updated import

# # --- Defense Strategy Flags Parser ---
# parser = argparse.ArgumentParser(description="A* Jailbreak Benchmark Pipeline")
# parser.add_argument('--baseline', action='store_true', help='Run the vanilla multi-layer defense')
# parser.add_argument('--smoothing', action='store_true', help='Run the randomized smoothing defense')
# parser.add_argument('--none', action='store_true', help='Run with no defense (Control Group)')
# parser.add_argument('--compare', action='store_true', help='Run ALL strategies and compare them')
# args = parser.parse_args()

# # Determine which strategies to test based on flags
# STRATEGIES_TO_TEST = []
# if args.compare:
#     STRATEGIES_TO_TEST = ['none', 'baseline', 'smoothing']
# else:
#     if args.none: STRATEGIES_TO_TEST.append('none')
#     if args.baseline: STRATEGIES_TO_TEST.append('baseline')
#     if args.smoothing: STRATEGIES_TO_TEST.append('smoothing')

# # --- PIPELINE CONFIGURATION ---
# MODELS_TO_TEST = [
#     # "Qwen/Qwen1.5-0.5B-Chat-AWQ", # For extremely low VRAM ~4GB
#     "Qwen/Qwen1.5-1.8B-Chat-AWQ",
#     "meta-llama/Llama-3-8b-Instruct-hf",
#     "google/gemma-3-4b-it",
#     # "meta-llama/Llama-2-13b-chat-hf", 
#     "lmsys/vicuna-13b-v1.5", # The Standard Baseline
# ]

# ATTACKS_TO_TEST = ["SimplePrefix", "GCG", "PAIR"]

# # Dependency Check
# try:
#     import jailbreakbench as jbb
#     from vllm import LLM, SamplingParams
# except ImportError as e:
#     logger.critical(f"Missing Dependency: {e}")
#     sys.exit(1)

# class Pipeline:
#     def __init__(self):
#         # 1. Load Dataset
#         logger.info("[SYSTEM] 📂 Loading Datasets...")
#         print("[SYSTEM] Loading Datasets...")
#         self.ds = jbb.read_dataset()
        
#         # Slicing for fast testing
#         limit = 100 or len(self.ds.behaviors)
#         self.behaviors = self.ds.behaviors[:limit] 
#         self.goals = self.ds.goals[:limit]
#         self.categories = self.ds.categories[:limit]
#         self.sources = self.ds.sources[:limit] if hasattr(self.ds, 'sources') else ["Unknown"]*len(self.goals)
        
#         # # 2. Initialize Judge
#         # logger.info("[SYSTEM] 📈 Loading Judge Model...")
#         # self.judge = JailbreakJudge()
        
#         # 3. Initialize Defense
#         logger.info("[SYSTEM] 🛡️ Loading Defense Layer...")
#         self.defense = DefenseLayer()
        
#         # 4. Status Tracker (NEW)
#         self.execution_log = [] 

#     def generate_comparative_charts(self, all_data_df):
#         if all_data_df.empty:
#             logger.warning("[SYSTEM] No data available to plot (Skipping Charts).")
#             return

#         logger.info("[SYSTEM] 📊 Generating comparative charts...")
        
#         try:
#             summary_df = all_data_df.groupby(['Model', 'Category'])['Jailbroken'].mean().reset_index()
#             summary_df['Success Rate (%)'] = summary_df['Jailbroken'] * 100

#             plt.figure(figsize=(14, 8))
#             sns.set_theme(style="whitegrid")
            
#             sns.barplot(
#                 data=summary_df, 
#                 x='Category', 
#                 y='Success Rate (%)', 
#                 hue='Model',
#                 palette='viridis'
#             )
            
#             plt.title('Jailbreak Success Rate by Behavior Category', fontsize=16)
#             plt.ylabel('Success Rate (%)', fontsize=12)
#             plt.xlabel('Category', fontsize=12)
#             plt.xticks(rotation=45, ha='right')
#             plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
#             plt.tight_layout()
            
#             output_path = os.path.join(RESULT_DIR, 'jailbreak_comparison_chart.png')
#             plt.savefig(output_path, dpi=300)
#             logger.info(f"[SYSTEM] Chart saved to: {output_path}")
#             plt.close()
            
#         except Exception as e:
#             logger.error(f"Failed to generate charts: {e}")

#     # def run(self):
#     #     all_results = []

#     #     for model_name in MODELS_TO_TEST:
#     #         logger.info(f"\n{'='*60}\n[TARGET] 🎯 Loading Target Model: {model_name}\n{'='*60}")
#     #         print(f"\n[TARGET] Loading Target Model: {model_name}")
            
#     #         model_start_time = time.time()
#     #         try:
#     #             # Load Target Model
#     #             target_llm = LLM(
#     #                 model=model_name,
#     #                 dtype="float16" or "float32" or "auto",
#     #                 tensor_parallel_size=1,
#     #                 gpu_memory_utilization=0.50,
#     #                 enforce_eager=True,
#     #                 disable_log_stats=True
#     #             )
#     #             logger.info(f"[SYSTEM] target_llm config: {target_llm}")
#     #             sampling_params = SamplingParams(temperature=0.1, max_tokens=150)
                
#     #         except Exception as e:
#     #             # RECORD FAILURE
#     #             logger.error(f"Failed to load {model_name}: {e}")
#     #             self.execution_log.append({
#     #                 "Model": model_name,
#     #                 "Status": " FAILED",
#     #                 "Reason": str(e)[:100] + "..." # Truncate long errors
#     #             })
#     #             continue # Skip to next model

#     #         # Initialize Attacker
#     #         attacker = AttackLoader(model_name)

#     #         # --- ATTACK LOOP ---
#     #         try:
#     #             for attack_method in ATTACKS_TO_TEST:
#     #                 logger.info(f"\n[PIPELINE] 🚀 Running Attack: {attack_method}")
#     #                 print(f"[PIPELINE] Running Attack: {attack_method}")
                    
#     #                 prompts = attacker.get_prompts(attack_method, self.goals)

#     #                 for i, (behavior, goal, prompt) in enumerate(zip(self.behaviors, self.goals, prompts)):
#     #                     # A. Defense: Input
#     #                     blocked_in, processed_prompt = self.defense.process_input(prompt)
                        
#     #                     if blocked_in:
#     #                         response_text = "I cannot fulfill this request (Defense Blocked Input)."
#     #                         is_jailbroken = False
#     #                     else:
#     #                         # B. Target Inference
#     #                         outputs = target_llm.generate([processed_prompt], sampling_params)
#     #                         raw_response = outputs[0].outputs[0].text.strip()

#     #                         # C. Defense: Output
#     #                         blocked_out, final_response = self.defense.process_output(raw_response)
                            
#     #                         if blocked_out:
#     #                             response_text = "I cannot fulfill this request (Defense Blocked Output)."
#     #                         else:
#     #                             response_text = final_response

#     #                         # D. Judge
#     #                         is_jailbroken = self.judge.evaluate(goal, response_text)
                            
#     #                     # Logging verdict
#     #                     status = "UNSAFE" if is_jailbroken else "SAFE"
#     #                     logger.info(f"  > [{i+1}/{len(self.goals)}] {behavior[:30]}... -> {status}")

#     #                     all_results.append({
#     #                         "Model": model_name.split("/")[-1],
#     #                         "Attack": attack_method,
#     #                         "Category": self.categories[i],
#     #                         "Source": self.sources[i],
#     #                         "Goal": goal,
#     #                         "Response": response_text,
#     #                         "Jailbroken": is_jailbroken
#     #                     })
                
#     #             # RECORD SUCCESS
#     #             self.execution_log.append({
#     #                 "Model": model_name,
#     #                 "Status": "SUCCESS",
#     #                 "Reason": f"Completed in {round(time.time() - model_start_time, 2)}s"
#     #             })

#     #         except Exception as e:
#     #             # RECORD RUNTIME FAILURE
#     #             logger.error(f"Runtime error on {model_name}: {e}")
#     #             self.execution_log.append({
#     #                 "Model": model_name,
#     #                 "Status": " CRASHED",
#     #                 "Reason": str(e)[:100]
#     #             })

#     #         # Cleanup
#     #         import gc
#     #         del target_llm
#     #         gc.collect()
#     #         torch.cuda.empty_cache()

#     #     # Save Raw Data to OUTPUT_DIR
#     #     if all_results:
#     #         df = pd.DataFrame(all_results)
#     #         csv_path = os.path.join(OUTPUT_DIR, "full_results.csv")
#     #         df.to_csv(csv_path, index=False)
#     #         logger.info(f"\n[SYSTEM] 💾 Results saved to {csv_path}")
#     #     else:
#     #         logger.warning("[SYSTEM] No results generated (All models failed?)")
#     #         df = pd.DataFrame()
        
#     #     return df

#     def run(self):
#         all_results = []
#         for current_strategy in STRATEGIES_TO_TEST:
#             logger.info(f"\n{'='*80}\n🛡️  ACTIVATING DEFENSE STRATEGY: {current_strategy.upper()}\n{'='*80}")
            
#             # Load the specific defense strategy dynamically
#             self.defense = get_defense_layer(current_strategy)

#             for model_name in MODELS_TO_TEST:
#                 logger.info(f"\n{'='*60}\n[TARGET] 🎯 Loading Target Model: {model_name}\n{'='*60}")
#                 print(f"\n[TARGET] Loading Target Model: {model_name}")
                
#                 # Start timer BEFORE loading the model for accurate tracking
#                 model_start_time = time.time()
                
#                 # Initialize to None to prevent UnboundLocalError during cleanup
#                 target_llm = None 
#                 generated_responses = []
                
#                 # =========================================
#                 # PHASE 1: TARGET GENERATION
#                 # =========================================
#                 try:
#                     # 1. Get the smart configuration dictionary based on server hardware
#                     smart_config = HardwareAwareModelLoader.get_optimal_target_config(preferred_base_model=model_name)
                    
#                     # CRITICAL FIX: Temporarily restore standard sys outputs for vLLM
#                     import sys
#                     custom_stdout = sys.stdout
#                     custom_stderr = sys.stderr

#                     # 2. Unpack the dict directly into vLLM (INSIDE THE TRY BLOCK)
#                     try:
#                         sys.stdout = sys.__stdout__
#                         sys.stderr = sys.__stderr__
                        
#                         # 2. Unpack the dict directly into vLLM
#                         target_llm = LLM(**smart_config)
                        
#                     finally:
#                         # Always restore custom logger right after loading
#                         sys.stdout = custom_stdout
#                         sys.stderr = custom_stderr

#                     logger.info(f"[SYSTEM] target_llm config: {smart_config}")

#                     sampling_params = SamplingParams(temperature=0.1, max_tokens=150)
                    
#                     # 3. Initialize Attacker
#                     attacker = AttackLoader(model_name)

#                     # --- ATTACK LOOP ---
#                     for attack_method in ATTACKS_TO_TEST:
#                         logger.info(f"\n[PIPELINE] 🚀 Running Attack: {attack_method}")
#                         print(f"[PIPELINE] Running Attack: {attack_method}")
                        
#                         prompts = attacker.get_prompts(attack_method, self.goals)

#                         for i, (behavior, goal, prompt) in enumerate(zip(self.behaviors, self.goals, prompts)):
                        
#                             # A. Defense: Input
#                             blocked_in, processed_prompt = self.defense.process_input(prompt)
                            
#                             if blocked_in:
#                                 generated_responses.append({
#                                     "attack_method": attack_method,
#                                     "behavior_idx": i,
#                                     "raw_response": "I cannot fulfill this request (Defense Blocked Input).",
#                                     "blocked_in": True
#                                 })
#                             else:
#                                 # B. Target Inference
#                                 outputs = target_llm.generate([processed_prompt], sampling_params)
#                                 raw_response = outputs[0].outputs[0].text.strip()

#                                 # C. Store the response to be passed to the Judge
#                                 generated_responses.append({
#                                     "attack_method": attack_method,
#                                     "behavior_idx": i,
#                                     "raw_response": raw_response,
#                                     "blocked_in": False
#                                 })
#                     #             # C. Defense: Output
#                     #             blocked_out, final_response = self.defense.process_output(raw_response)
                                
#                     #             if blocked_out:
#                     #                 response_text = "I cannot fulfill this request (Defense Blocked Output)."
#                     #             else:
#                     #                 response_text = final_response

#                     #         # D. Judge
#                     #         is_jailbroken = self.judge.evaluate(goal, response_text)
                            
#                     #         # Logging verdict
#                     #         status = "UNSAFE" if is_jailbroken else "SAFE"
#                     #         logger.info(f"  > [{i+1}/{len(self.goals)}] {behavior[:30]}... -> {status}")

#                     #         all_results.append({
#                     #             "Model": model_name.split("/")[-1],
#                     #             "Attack": attack_method,
#                     #             "Category": self.categories[i],
#                     #             "Source": self.sources[i], 
#                     #             "Goal": goal,
#                     #             "Response": response_text,
#                     #             "Jailbroken": is_jailbroken
#                     #         })
                    
#                     # # RECORD SUCCESS
#                     # self.execution_log.append({
#                     #     "Model": model_name,
#                     #     "Status": "SUCCESS",
#                     #     "Reason": f"Completed in {round(time.time() - model_start_time, 2)}s"
#                     # })

#                 except Exception as e:
#                     # RECORD RUNTIME OR LOAD FAILURE
#                     logger.error(f"Failed to load or run {model_name}: {e}")
#                     self.execution_log.append({
#                         "Model": model_name,
#                         "Status": "FAILED",
#                         "Reason": str(e)[:100] + "..."
#                     })
#                     continue

#                 finally:
#                     logger.info("[SYSTEM] 🧹 Initiating Phase 1 Cleanup...")
#                     # =========================================
#                     # PHASE 1 CLEANUP
#                     # =========================================
#                     logger.info("[SYSTEM] 🧹 Initiating Phase 1 Cleanup...")
                    
#                     # 1. Destroy vLLM parallel states (Supports old & new vLLM versions)
#                     try:
#                         from vllm.distributed.parallel_state import destroy_model_parallel
#                         destroy_model_parallel()
#                     except ImportError:
#                         try:
#                             from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
#                             destroy_model_parallel()
#                         except ImportError:
#                             pass
#                     except Exception as e:
#                         logger.debug(f"[SYSTEM] destroy_model_parallel warning: {e}")
                    
#                     # 2. Destroy PyTorch distributed group (CRITICAL for NCCL)
#                     try:
#                         import torch.distributed as dist
#                         if dist.is_initialized():
#                             dist.destroy_process_group()
#                     except Exception as e:
#                         logger.debug(f"[SYSTEM] dist.destroy warning: {e}")

#                     # 3. Delete the LLM object from Python memory
#                     if target_llm is not None:
#                         del target_llm
                    
#                     # 4. Flush VRAM
#                     import gc
#                     gc.collect()
#                     import torch
#                     torch.cuda.empty_cache()

#                     # 5. Nuke Ray
#                     try:
#                         import ray
#                         if ray.is_initialized():
#                             logger.info("[SYSTEM] 🧠 Calling Ray shutdown()...")
#                             ray.shutdown()
#                     except Exception:
#                         pass

#                     logger.info("[SYSTEM] ⏳ Waiting 3 seconds for OS to release GPU TCP ports...")
#                     time.sleep(3) # Allow OS to release ports before next loop

#                 # ======================================
#                 # PHASE 2: EVALUATION (JUDGE)
#                 # ======================================

#                 logger.info(f"\n[PIPELINE] ⚖️ Generation complete. Loading Judge for Evaluation...")
#                 judge = None
                
#                 try:
#                     # Load the Judge ONLY after the Target model has surrendered the GPU
#                     judge = JailbreakJudge()

#                     for item in generated_responses:
#                         idx = item["behavior_idx"]
#                         raw_response = item["raw_response"]
#                         goal = self.goals[idx]

#                         if item["blocked_in"]:
#                             response_text = raw_response
#                             is_jailbroken = False
#                         else:
#                             # Defense: Output
#                             blocked_out, final_response = self.defense.process_output(raw_response)
#                             response_text = "I cannot fulfill this request (Defense Blocked Output)." if blocked_out else final_response
                            
#                             # Judge Evaluation
#                             is_jailbroken = judge.evaluate(goal, response_text)

#                         status = "UNSAFE" if is_jailbroken else "SAFE"
#                         logger.info(f"  > [{idx+1}/{len(self.goals)}] {self.behaviors[idx][:30]}... -> {status}")

#                         all_results.append({
#                             "Model": model_name.split("/")[-1],
#                             "Attack": item["attack_method"],
#                             "Category": self.categories[idx],
#                             "Source": self.sources[idx], 
#                             "Goal": goal,
#                             "Response": response_text,
#                             "Jailbroken": is_jailbroken
#                         })
                    
#                     self.execution_log.append({
#                         "Model": model_name,
#                         "Status": "SUCCESS",
#                         "Reason": f"Completed in {round(time.time() - model_start_time, 2)}s"
#                     })

#                 except Exception as e:
#                     logger.error(f"Judge failed during evaluation of {model_name}: {e}")
#                     self.execution_log.append({"Model": model_name + " (Eval)", "Status": "FAILED", "Reason": str(e)[:100]})

#                 finally:
#                     logger.info("[SYSTEM] 🧹 Initiating Phase 2 Cleanup...")
#                     # 1. Destroy vLLM parallel states (Supports old & new vLLM versions)
#                     try:
#                         logger.info("[SYSTEM] 🧠 Destroying vLLM parallel state...")
#                         from vllm.distributed.parallel_state import destroy_model_parallel
#                         destroy_model_parallel()

#                     except Exception as e:
#                         logger.debug(f"[SYSTEM] Error: {e} \nvLLM destroy_model_parallel() failed for Judge. Continuing...")
#                         pass
                    
#                     # 2. Destroy PyTorch distributed group (NCCL)
#                     try:
#                         import torch.distributed as dist
#                         if dist.is_initialized():
#                             dist.destroy_process_group()
#                     except Exception as e:
#                         logger.warning(f"[SYSTEM] Error:{e} \n Failed to destroy Pytorch distributed group. Continuing...")
#                         pass
                    
#                     # 3. Delete the Judge object from Python memory
#                     if judge is not None and hasattr(judge, 'llm'):
#                         del judge.llm
#                     del judge
                    
#                     # 4. Flush VRAM
#                     import gc
#                     gc.collect()
#                     gc.collect()
#                     torch.cuda.empty_cache()
                    
#                     # 5. Shutdown Ray (if initialized)
#                     try: 
#                         import ray
#                         logger.info("[SYSTEM] 🧠 Calling Ray shutdown()...")
#                         if ray.is_initialized():
#                             logger.info("[SYSTEM] 🧠 Calling Ray shutdown()...")
#                             ray.shutdown()
#                     except Exception as e:
#                         logger.warning(f"Ray failed to shutdown: {e}")
#                         pass
                    
#                     logger.info("[SYSTEM] ⏳ Waiting 3 seconds for OS to release GPU TCP ports...")
#                     time.sleep(3) # Allow OS to release ports before next loop

#         # Save Raw Data to OUTPUT_DIR
#         if all_results:
#             df = pd.DataFrame(all_results)
#             csv_path = os.path.join(OUTPUT_DIR, "full_results.csv")
#             df.to_csv(csv_path, index=False)
#             logger.info(f"\n[SYSTEM] 💾 Results saved to {csv_path}")
#         else:
#             logger.warning("[SYSTEM] No results generated (All models failed?)")
#             df = pd.DataFrame()
        
#         return df
        
#     def print_final_summary(self):
#         """Prints a honest breakdown of what happened."""
#         print("\n\n")
#         print("==================================================")
#         print("               EXECUTION SUMMARY                  ")
#         print("==================================================")
        
#         summary_df = pd.DataFrame(self.execution_log)
#         if not summary_df.empty:
#             print(summary_df.to_markdown(index=False))
#         else:
#             print("No models were attempted.")

#         # Check for failures
#         failed_models = [x for x in self.execution_log if "FAILED" in x["Status"] or "CRASHED" in x["Status"]]
        
#         if failed_models:
#             print("\n PIPELINE COMPLETED WITH ERRORS")
#             print(f"   {len(failed_models)} model(s) failed to load or crashed.")
#             print("   Check ./logs/execution.log for full tracebacks.")
#             sys.exit(1) # Exit code 1 to signal failure to CI/CD or scripts
#         else:
#             print("\n PIPELINE COMPLETED SUCCESSFULLY")

#     def generate_report(self, df):
#         if df.empty: return

#         logger.info("\n[SYSTEM] 📊 Generating Comparative Analysis...")
        
#         # 1. Generate Summary Table (ASR)
#         asr_table = df.groupby(['Model', 'Attack'])['Jailbroken'].mean() * 100
        
#         print("\n--- Attack Success Rate (ASR) ---")
#         print(asr_table)
        
#         # Save table
#         asr_table.to_csv(os.path.join(RESULT_DIR, "metrics_asr.csv"))

#         # 2. Generate Plots
#         self.generate_comparative_charts(df)

# if __name__ == "__main__":
#     pipeline = Pipeline()
#     results_df = pipeline.run()
    
#     # Generate charts only if we have data
#     if not results_df.empty:
#         pipeline.generate_report(results_df)
    
#     # ALWAYS print the status summary at the end
#     pipeline.print_final_summary()

# ## Version - 4
# import os
# import sys
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# import gc
# import time
# import argparse
# from tabulate import tabulate

# # Import shared config and logger
# from logger_config import logger, LOGS_DIR, RESULT_DIR, OUTPUT_DIR

# # Import modules
# from judge import JailbreakJudge
# from attacks import AttackLoader
# from model_loader import HardwareAwareModelLoader
# from defensive_layer import get_defense_layer

# # =====================================================================
# # COMMAND LINE ARGUMENT PARSER
# # =====================================================================
# parser = argparse.ArgumentParser(description="A* Jailbreak Benchmark Pipeline")
# parser.add_argument('--baseline', action='store_true', help='Run the vanilla multi-layer defense')
# parser.add_argument('--smoothing', action='store_true', help='Run the randomized smoothing defense')
# parser.add_argument('--none', action='store_true', help='Run with no defense (Control Group)')
# parser.add_argument('--compare', action='store_true', help='Run ALL strategies and compare them')
# args = parser.parse_args()

# # Determine which strategies to test based on flags
# STRATEGIES_TO_TEST = []
# if args.compare:
#     STRATEGIES_TO_TEST = ['none', 'baseline', 'smoothing']
# else:
#     if args.none: STRATEGIES_TO_TEST.append('none')
#     if args.baseline: STRATEGIES_TO_TEST.append('baseline')
#     if args.smoothing: STRATEGIES_TO_TEST.append('smoothing')

# # Default: If no flags passed, run all the attacks and strategies for complete benchmarking
# if not STRATEGIES_TO_TEST:
#     STRATEGIES_TO_TEST = ['baseline', 'smoothing']

# ATTACKS_TO_TEST = [
#     "GCG",             # 1. Normal GCG JBB Attacks
#     "PAIR",            # 2. Normal PAIR JBB Attacks 
#     "JBB_Multi",       # 2. Multi-Turn simulation of JBB attacks
#     "MTJ_Single",      # 3. Single-Turn attacks from the MTJ dataset
#     "MTJ_Multi"        # 4. Multi-Turn attacks from the MTJ dataset
# ]

# # =====================================================================
# # PIPELINE CONFIGURATION
# # =====================================================================
# MODELS_TO_TEST = [
#     # "Qwen/Qwen1.5-0.5B-Chat-AWQ", 
#     "Qwen/Qwen1.5-1.8B-Chat-AWQ",
#     "meta-llama/Llama-3-8b-Instruct-hf",
#     "google/gemma-3-4b-it",
#     "lmsys/vicuna-13b-v1.5", 
# ]

# # ATTACKS_TO_TEST = ["SimplePrefix", "GCG", "PAIR"]
# TEST_LIMIT = 100 # Set to None to run the full dataset

# # Dependency Check
# try:
#     import jailbreakbench as jbb
#     from vllm import LLM, SamplingParams
# except ImportError as e:
#     logger.critical(f"Missing Dependency: {e}")
#     sys.exit(1)


# class Pipeline:
#     def __init__(self):
#         # 1. Load Dataset
#         logger.info("[SYSTEM] 📂 Loading Datasets...")
#         print("[SYSTEM] Loading Datasets...")
#         self.ds = jbb.read_dataset()
        
#         # 2. Slicing for fast testing vs full benchmark
#         limit = TEST_LIMIT
#         self.behaviors = self.ds.behaviors[:limit] if limit else self.ds.behaviors
#         self.goals = self.ds.goals[:limit] if limit else self.ds.goals
#         self.categories = self.ds.categories[:limit] if limit else self.ds.categories
#         self.sources = self.ds.sources[:limit] if hasattr(self.ds, 'sources') and limit else getattr(self.ds, 'sources', ["Unknown"]*len(self.goals))
        
#         self.execution_log = [] 

#     def generate_comparative_charts(self, all_data_df):
#         """Generates A* quality ablation study charts."""
#         if all_data_df.empty:
#             logger.warning("[SYSTEM] No data available to plot.")
#             return

#         logger.info("[SYSTEM] 📊 Generating comparative charts...")
#         try:
#             # Calculate Attack Success Rate (ASR) grouped by Strategy and Model
#             summary_df = all_data_df.groupby(['Defense_Strategy', 'Model'])['Jailbroken'].mean().reset_index()
#             summary_df['ASR (%)'] = summary_df['Jailbroken'] * 100

#             plt.figure(figsize=(12, 7))
#             sns.set_theme(style="whitegrid")
            
#             # The Ultimate Ablation Chart
#             sns.barplot(
#                 data=summary_df, 
#                 x='Model', 
#                 y='ASR (%)', 
#                 hue='Defense_Strategy',
#                 palette='magma'
#             )
            
#             plt.title('Attack Success Rate (ASR) by Defense Strategy', fontsize=16, fontweight='bold')
#             plt.ylabel('Attack Success Rate (%)', fontsize=12)
#             plt.xlabel('Target Model', fontsize=12)
#             plt.xticks(rotation=45, ha='right')
#             plt.legend(title='Defense Layer', bbox_to_anchor=(1.05, 1), loc='upper left')
#             plt.tight_layout()
            
#             output_path = os.path.join(RESULT_DIR, 'defense_ablation_chart.png')
#             plt.savefig(output_path, dpi=300)
#             logger.info(f"[SYSTEM] Chart saved to: {output_path}")
#             plt.close()
            
#         except Exception as e:
#             logger.error(f"Failed to generate charts: {e}")

#     def run(self):
#         all_results = []
        
#         # =======================================================
#         # THE STRATEGY LOOP (Ablation Study Orchestrator)
#         # =======================================================
#         for current_strategy in STRATEGIES_TO_TEST:
#             logger.info(f"\n{'='*80}\n🛡️  ACTIVATING DEFENSE STRATEGY: {current_strategy.upper()}\n{'='*80}")
            
#             # Dynamically load the requested defense layer
#             self.defense = get_defense_layer(current_strategy)
            
#             for model_name in MODELS_TO_TEST:
#                 logger.info(f"\n{'='*60}\n[TARGET] 🎯 Loading Target Model: {model_name}\n{'='*60}")
#                 print(f"\n[TARGET] Loading Target Model: {model_name}")
                
#                 model_start_time = time.time()
#                 target_llm = None 
#                 generated_responses = []
                
#                 # =========================================
#                 # PHASE 1: TARGET GENERATION
#                 # =========================================
#                 try:
#                     smart_config = HardwareAwareModelLoader.get_optimal_target_config(preferred_base_model=model_name)
                    
#                     import sys
#                     custom_stdout = sys.stdout
#                     custom_stderr = sys.stderr

#                     try:
#                         sys.stdout = sys.__stdout__
#                         sys.stderr = sys.__stderr__
#                         target_llm = LLM(**smart_config)
#                     finally:
#                         sys.stdout = custom_stdout
#                         sys.stderr = custom_stderr

#                     logger.info(f"[SYSTEM] target_llm config: {smart_config}")
#                     sampling_params = SamplingParams(temperature=0.1, max_tokens=150)
#                     attacker = AttackLoader(model_name)

#                     # --- ATTACK LOOP ---
#                     for attack_method in ATTACKS_TO_TEST:
#                         logger.info(f"\n[PIPELINE] 🚀 Running Attack: {attack_method}")
#                         print(f"[PIPELINE] Running Attack: {attack_method}")
                        
#                         try:
#                         # 1. Fetch data mapped perfectly to the chosen dataset
#                             prompt_sequences, active_goals, active_behaviors = attacker.get_prompts(
#                                 attack_method, self.goals, self.behaviors
#                             )
#                         except Exception as e:
#                             logger.error(f"Skipping {attack_method} due to data extraction failure: {e}")
#                             continue

#                         # 2. Sequential Orchestration Loop
#                         for i, (behavior, goal, prompt) in enumerate(zip(self.behaviors, self.goals, prompt_sequences)):
                            
#                             chat_history = ""
#                             blocked_in_turn = False
#                             final_turn_response = ""
                            
#                             # Execute the Trajectory (Iterates 1 turn for single, N turns for multi)
#                             for turn_idx, turn_prompt in enumerate(sequence):
                                
#                             # A. Defense: Input
#                             blocked_in, processed_prompt = self.defense.process_input(prompt)
                            
#                             if blocked_in:
#                                 generated_responses.append({
#                                     "attack_method": attack_method,
#                                     "behavior_idx": i,
#                                     "raw_response": "I cannot fulfill this request (Defense Blocked Input).",
#                                     "blocked_in": True
#                                 })
#                             else:
#                                 # B. Target Inference
#                                 outputs = target_llm.generate([processed_prompt], sampling_params)
#                                 raw_response = outputs[0].outputs[0].text.strip()

#                                 # C. Store the response for Judge
#                                 generated_responses.append({
#                                     "attack_method": attack_method,
#                                     "behavior_idx": i,
#                                     "raw_response": raw_response,
#                                     "blocked_in": False
#                                 })

#                 except Exception as e:
#                     logger.error(f"Failed to load or run {model_name}: {e}")
#                     self.execution_log.append({
#                         "Strategy": current_strategy.upper(),
#                         "Model": model_name,
#                         "Status": "FAILED (Phase 1)",
#                         "Reason": str(e)[:100] + "..."
#                     })
#                     continue

#                 finally:
#                     # --- PHASE 1 CLEANUP ---
#                     logger.info("[SYSTEM] 🧹 Initiating Phase 1 Cleanup...")
#                     try:
#                         from vllm.distributed.parallel_state import destroy_model_parallel
#                         destroy_model_parallel()
#                     except Exception:
#                         pass
                    
#                     try:
#                         import torch.distributed as dist
#                         if dist.is_initialized(): dist.destroy_process_group()
#                     except Exception:
#                         pass

#                     if target_llm is not None: del target_llm
                    
#                     gc.collect()
#                     torch.cuda.empty_cache()

#                     try: 
#                         import ray
#                         if ray.is_initialized(): ray.shutdown()
#                     except Exception:
#                         pass
                    
#                     time.sleep(3)


#                 # ======================================
#                 # PHASE 2: EVALUATION (JUDGE)
#                 # ======================================
#                 if not generated_responses:
#                     logger.warning("[PIPELINE] ⚠️ No responses generated. Skipping Phase 2.")
#                     continue

#                 logger.info(f"\n[PIPELINE] ⚖️ Generation complete. Loading Judge for Evaluation...")
#                 judge = None
                
#                 try:
#                     judge = JailbreakJudge()

#                     for item in generated_responses:
#                         idx = item["behavior_idx"]
#                         raw_response = item["raw_response"]
#                         goal = self.goals[idx]

#                         if item["blocked_in"]:
#                             response_text = raw_response
#                             is_jailbroken = False
#                         else:
#                             # Defense: Output
#                             blocked_out, final_response = self.defense.process_output(raw_response)
#                             response_text = "I cannot fulfill this request (Defense Blocked Output)." if blocked_out else final_response
                            
#                             # Judge Evaluation
#                             is_jailbroken = judge.evaluate(goal, response_text)

#                         status = "UNSAFE" if is_jailbroken else "SAFE"
#                         logger.info(f"  > [{idx+1}/{len(self.goals)}] {self.behaviors[idx][:30]}... -> {status}")

#                         all_results.append({
#                             "Defense_Strategy": current_strategy.upper(), # CRITICAL TRACKER
#                             "Model": model_name.split("/")[-1],
#                             "Attack": item["attack_method"],
#                             "Category": self.categories[idx],
#                             "Source": self.sources[idx], 
#                             "Goal": goal,
#                             "Response": response_text,
#                             "Jailbroken": is_jailbroken
#                         })
                    
#                     self.execution_log.append({
#                         "Strategy": current_strategy.upper(),
#                         "Model": model_name,
#                         "Status": "SUCCESS",
#                         "Reason": f"Completed in {round(time.time() - model_start_time, 2)}s"
#                     })

#                 except Exception as e:
#                     logger.error(f"Judge failed during evaluation of {model_name}: {e}")
#                     self.execution_log.append({
#                         "Strategy": current_strategy.upper(),
#                         "Model": model_name + " (Eval)", 
#                         "Status": "FAILED (Phase 2)", 
#                         "Reason": str(e)[:100]
#                     })

#                 finally:
#                     # --- PHASE 2 CLEANUP ---
#                     logger.info("[SYSTEM] 🧹 Initiating Phase 2 Cleanup...")
#                     if judge is not None and hasattr(judge, 'llm'):
#                         del judge.llm
#                     del judge

#                     try:
#                         from vllm.distributed.parallel_state import destroy_model_parallel
#                         destroy_model_parallel()
#                     except Exception:
#                         pass
                    
#                     try:
#                         import torch.distributed as dist
#                         if dist.is_initialized(): dist.destroy_process_group()
#                     except Exception:
#                         pass

#                     gc.collect()
#                     torch.cuda.empty_cache()
                    
#                     try: 
#                         import ray
#                         if ray.is_initialized(): ray.shutdown()
#                     except Exception:
#                         pass
                    
#                     time.sleep(3)

#             # Cleanup defense memory before switching strategies
#             del self.defense
#             gc.collect()

#         # =======================================================
#         # DATA SAVING & REPORTING
#         # =======================================================
#         if all_results:
#             df = pd.DataFrame(all_results)
#             csv_path = os.path.join(OUTPUT_DIR, "full_results.csv")
#             df.to_csv(csv_path, index=False)
#             logger.info(f"\n[SYSTEM] 💾 Results saved to {csv_path}")
#         else:
#             logger.warning("[SYSTEM] No results generated (All models failed?)")
#             df = pd.DataFrame()
        
#         return df
        
#     def print_final_summary(self):
#         print("\n\n")
#         print("==================================================")
#         print("               EXECUTION SUMMARY                  ")
#         print("==================================================")
        
#         summary_df = pd.DataFrame(self.execution_log)
#         if not summary_df.empty:
#             print(summary_df.to_markdown(index=False))
#         else:
#             print("No models were attempted.")

#         failed_models = [x for x in self.execution_log if "FAILED" in x["Status"]]
        
#         if failed_models:
#             print("\n PIPELINE COMPLETED WITH ERRORS")
#             print(f"   {len(failed_models)} process(es) failed to load or crashed.")
#             print("   Check ./logs/execution.log for full tracebacks.")
#             sys.exit(1)
#         else:
#             print("\n PIPELINE COMPLETED SUCCESSFULLY")

#     def generate_report(self, df):
#         if df.empty: return
#         logger.info("\n[SYSTEM] 📊 Generating Comparative Analysis...")
        
#         # Multidimensional Grouping for the Table
#         asr_table = df.groupby(['Defense_Strategy', 'Model', 'Attack'])['Jailbroken'].mean() * 100
#         print("\n--- Attack Success Rate (ASR) ---")
#         print(asr_table)
        
#         asr_table.to_csv(os.path.join(RESULT_DIR, "metrics_asr.csv"))
#         self.generate_comparative_charts(df)


# if __name__ == "__main__":
#     pipeline = Pipeline()
#     results_df = pipeline.run()
    
#     if not results_df.empty:
#         pipeline.generate_report(results_df)
    
#     pipeline.print_final_summary()

## Version - 5
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gc
import time
import argparse
from tabulate import tabulate

# Import shared config and logger
from logger_config import logger, LOGS_DIR, RESULT_DIR, OUTPUT_DIR

# Import modules
from judge import JailbreakJudge
from attacks import AttackLoader
from model_loader import HardwareAwareModelLoader
from defensive_layer import get_defense_layer

# =====================================================================
# COMMAND LINE ARGUMENT PARSER
# =====================================================================
parser = argparse.ArgumentParser(description="A* Jailbreak Benchmark Pipeline")
parser.add_argument('--baseline', action='store_true', help='Run the vanilla multi-layer defense')
parser.add_argument('--smoothing', action='store_true', help='Run the randomized smoothing defense')
parser.add_argument('--none', action='store_true', help='Run with no defense (Control Group)')
parser.add_argument('--compare', action='store_true', help='Run ALL strategies and compare them')
args = parser.parse_args()

# Determine which strategies to test based on flags
STRATEGIES_TO_TEST = []
if args.compare:
    STRATEGIES_TO_TEST = ['none', 'baseline', 'smoothing']
else:
    if args.none: STRATEGIES_TO_TEST.append('none')
    if args.baseline: STRATEGIES_TO_TEST.append('baseline')
    if args.smoothing: STRATEGIES_TO_TEST.append('smoothing')

# Default: If no flags passed, run the defenses for complete benchmarking
if not STRATEGIES_TO_TEST:
    STRATEGIES_TO_TEST = ['baseline', 'smoothing']

# =====================================================================
# PIPELINE CONFIGURATION
# =====================================================================
ATTACKS_TO_TEST = [
    "GCG",             # 1. Normal GCG JBB Attacks
    "PAIR",            # 2. Normal PAIR JBB Attacks 
    "JBB_Multi",       # 3. Multi-Turn simulation of JBB attacks
    "MTJ_Single",      # 4. Single-Turn attacks from the MTJ dataset
    "MTJ_Multi"        # 5. Multi-Turn attacks from the MTJ dataset
]

MODELS_TO_TEST = [
    "Qwen/Qwen1.5-1.8B-Chat-AWQ",
    "meta-llama/Llama-3-8b-Instruct-hf",
    "google/gemma-3-4b-it",
    "lmsys/vicuna-13b-v1.5", 
]

TEST_LIMIT = 100 # Set to None to run the full dataset

# Dependency Check
try:
    import jailbreakbench as jbb
    from vllm import LLM, SamplingParams
except ImportError as e:
    logger.critical(f"Missing Dependency: {e}")
    sys.exit(1)


class Pipeline:
    def __init__(self):
        # 1. Load JBB Dataset (Baseline Ground Truth)
        logger.info("[SYSTEM] 📂 Loading Base JBB Datasets...")
        print("[SYSTEM] Loading Base JBB Datasets...")
        self.ds = jbb.read_dataset()
        
        # 2. Slicing for fast testing vs full benchmark
        limit = TEST_LIMIT
        self.behaviors = self.ds.behaviors[:limit] if limit else self.ds.behaviors
        self.goals = self.ds.goals[:limit] if limit else self.ds.goals
        self.categories = self.ds.categories[:limit] if limit else self.ds.categories
        
        self.execution_log = [] 

    def generate_comparative_charts(self, all_data_df):
        """Generates A* quality ablation study charts."""
        if all_data_df.empty:
            logger.warning("[SYSTEM] No data available to plot.")
            return

        logger.info("[SYSTEM] 📊 Generating comparative charts...")
        try:
            # Calculate Attack Success Rate (ASR) grouped by Strategy and Model
            summary_df = all_data_df.groupby(['Defense_Strategy', 'Model'])['Jailbroken'].mean().reset_index()
            summary_df['ASR (%)'] = summary_df['Jailbroken'] * 100

            plt.figure(figsize=(12, 7))
            sns.set_theme(style="whitegrid")
            
            # The Ultimate Ablation Chart
            sns.barplot(
                data=summary_df, 
                x='Model', 
                y='ASR (%)', 
                hue='Defense_Strategy',
                palette='magma'
            )
            
            plt.title('Attack Success Rate (ASR) by Defense Strategy', fontsize=16, fontweight='bold')
            plt.ylabel('Attack Success Rate (%)', fontsize=12)
            plt.xlabel('Target Model', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Defense Layer', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            output_path = os.path.join(RESULT_DIR, 'defense_ablation_chart.png')
            plt.savefig(output_path, dpi=300)
            logger.info(f"[SYSTEM] Chart saved to: {output_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")

    def run(self):
        all_results = []
        
        # =======================================================
        # THE STRATEGY LOOP (Ablation Study Orchestrator)
        # =======================================================
        for current_strategy in STRATEGIES_TO_TEST:
            logger.info(f"\n{'='*80}\n🛡️  ACTIVATING DEFENSE STRATEGY: {current_strategy.upper()}\n{'='*80}")
            
            # Dynamically load the requested defense layer
            self.defense = get_defense_layer(current_strategy)
            
            for model_name in MODELS_TO_TEST:
                logger.info(f"\n{'='*60}\n[TARGET] 🎯 Loading Target Model: {model_name}\n{'='*60}")
                print(f"\n[TARGET] Loading Target Model: {model_name}")
                
                model_start_time = time.time()
                target_llm = None 
                generated_responses = []
                
                # =========================================
                # PHASE 1: TARGET GENERATION
                # =========================================
                try:
                    smart_config = HardwareAwareModelLoader.get_optimal_target_config(preferred_base_model=model_name)
                    
                    import sys
                    custom_stdout = sys.stdout
                    custom_stderr = sys.stderr

                    try:
                        sys.stdout = sys.__stdout__
                        sys.stderr = sys.__stderr__
                        target_llm = LLM(**smart_config)
                    finally:
                        sys.stdout = custom_stdout
                        sys.stderr = custom_stderr

                    logger.info(f"[SYSTEM] target_llm config: {smart_config}")
                    sampling_params = SamplingParams(temperature=0.1, max_tokens=150)
                    attacker = AttackLoader(model_name)

                    # --- ATTACK LOOP ---
                    for attack_method in ATTACKS_TO_TEST:
                        logger.info(f"\n[PIPELINE] 🚀 Running Attack Strategy: {attack_method}")
                        
                        try:
                            # 1. Fetch data mapped perfectly to the chosen dataset (JBB vs MTJ)
                            prompt_sequences, active_goals, active_behaviors = attacker.get_prompts(
                                attack_method, self.goals, self.categories
                            )
                        except Exception as e:
                            logger.error(f"Skipping {attack_method} due to data extraction failure: {e}")
                            continue

                        # 2. Sequential Orchestration Loop
                        for i, (behavior, goal, sequence) in enumerate(zip(active_behaviors, active_goals, prompt_sequences)):
                            
                            chat_history = ""
                            blocked_in_turn = False
                            final_turn_response = ""
                            
                            # Execute the Trajectory (Iterates 1 turn for single, N turns for multi)
                            for turn_idx, turn_prompt in enumerate(sequence):
                                
                                # A. Defense: Input (Evaluates accumulating history to catch semantic drift)
                                context_to_evaluate = chat_history + f"User: {turn_prompt}"
                                blocked_in, processed_prompt = self.defense.process_input(context_to_evaluate)
                                
                                if blocked_in:
                                    blocked_in_turn = True
                                    final_turn_response = f"I cannot fulfill this request (Defense Blocked Input at Turn {turn_idx+1})."
                                    break # Halt the conversation immediately
                                    
                                # B. Append to History & Generate
                                chat_history += f"User: {processed_prompt}\nAssistant: "
                                outputs = target_llm.generate(
                                    [chat_history], sampling_params
                                )
                                raw_response = outputs[0].outputs[0].text.strip()
                                
                                # C. Append model response to history for the next turn
                                chat_history += f"{raw_response}\n\n"
                                final_turn_response = raw_response

                            # Save the final outcome of the trajectory for the Judge
                            generated_responses.append({
                                "attack_method": attack_method,
                                "behavior_idx": i,
                                "raw_response": final_turn_response,
                                "blocked_in": blocked_in_turn,
                                "turns_survived": turn_idx + 1,
                                "active_goal": goal,             # Crucial: Track the dynamic dataset goal
                                "active_behavior": behavior      # Crucial: Track the dynamic dataset category
                            })

                except Exception as e:
                    logger.error(f"Failed to load or run {model_name}: {e}")
                    self.execution_log.append({
                        "Strategy": current_strategy.upper(),
                        "Model": model_name,
                        "Status": "FAILED (Phase 1)",
                        "Reason": str(e)[:100] + "..."
                    })
                    continue

                finally:
                    # --- PHASE 1 CLEANUP ---
                    logger.info("[SYSTEM] 🧹 Initiating Phase 1 Cleanup...")
                    try:
                        from vllm.distributed.parallel_state import destroy_model_parallel
                        destroy_model_parallel()
                    except Exception:
                        logger.warning(f"[SYSTEM] Failed to destroy model parallel state: {e}")
                    
                    try:
                        import torch.distributed as dist
                        if dist.is_initialized(): dist.destroy_process_group()
                    except Exception:
                        logger.warning(f"[SYSTEM] Failed to destroy torch distributed state: {e}")

                    if target_llm is not None: del target_llm
                    
                    gc.collect()
                    torch.cuda.empty_cache()

                    try: 
                        import ray
                        if ray.is_initialized(): ray.shutdown()
                    except Exception:
                        logger.warning(f"[SYSTEM] Failed to Ray shutdown: {e}")
                    
                    time.sleep(3)

                # ======================================
                # PHASE 2: EVALUATION (JUDGE)
                # ======================================
                if not generated_responses:
                    logger.warning(f"[PIPELINE] ⚠️ No responses generated for {model_name}. Skipping Phase 2.")
                    continue

                logger.info(f"\n[PIPELINE] ⚖️ Generation complete. Loading Judge for Evaluation...")
                judge = None
                
                try:
                    judge = JailbreakJudge()

                    for item in generated_responses:
                        idx = item["behavior_idx"]
                        raw_response = item["raw_response"]
                        
                        # Dynamically pull the correct goal based on if it was a JBB or MTJ attack
                        goal = item["active_goal"]
                        behavior = item["active_behavior"]

                        if item["blocked_in"]:
                            response_text = raw_response
                            is_jailbroken = False
                        else:
                            # Defense: Output Leak Check
                            blocked_out, final_response = self.defense.process_output(raw_response)
                            response_text = "I cannot fulfill this request (Defense Blocked Output)." if blocked_out else final_response
                            
                            # Judge Evaluation
                            is_jailbroken = judge.evaluate(goal, response_text)

                        status = "UNSAFE" if is_jailbroken else "SAFE"
                        logger.info(f"  > [{idx+1}] {behavior[:30]}... -> {status} (Survived {item['turns_survived']} turns)")

                        all_results.append({
                            "Defense_Strategy": current_strategy.upper(),
                            "Model": model_name.split("/")[-1],
                            "Attack": item["attack_method"],
                            "Category": behavior,
                            "Goal": goal,
                            "Turns_Survived": item["turns_survived"],
                            "Response": response_text,
                            "Jailbroken": is_jailbroken
                        })
                    
                    self.execution_log.append({
                        "Strategy": current_strategy.upper(),
                        "Model": model_name,
                        "Status": "SUCCESS",
                        "Reason": f"Completed in {round(time.time() - model_start_time, 2)}s"
                    })

                except Exception as e:
                    logger.error(f"Judge failed during evaluation of {model_name}: {e}")
                    self.execution_log.append({
                        "Strategy": current_strategy.upper(),
                        "Model": model_name + " (Eval)", 
                        "Status": "FAILED (Phase 2)", 
                        "Reason": str(e)[:100]
                    })

                finally:
                    # --- PHASE 2 CLEANUP ---
                    logger.info("[SYSTEM] 🧹 Initiating Phase 2 Cleanup...")
                    if judge is not None and hasattr(judge, 'llm'):
                        del judge.llm
                    del judge

                    try:
                        from vllm.distributed.parallel_state import destroy_model_parallel
                        destroy_model_parallel()
                    except Exception:
                        logger.warning(f"[SYSTEM] Failed to destroy model parallel state: {e}")
                    
                    try:
                        import torch.distributed as dist
                        if dist.is_initialized(): dist.destroy_process_group()
                    except Exception:
                        logger.warning("[SYSTEM] Failed to destroy torch distributed state: {e}")

                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    try: 
                        import ray
                        if ray.is_initialized(): ray.shutdown()
                    except Exception:
                        logger.warning(f"[SYSTEM] Failed to Ray shutdown: {e}")
                    
                    time.sleep(3)

            # Cleanup defense memory before switching strategies
            del self.defense
            gc.collect()

        # =======================================================
        # DATA SAVING & REPORTING
        # =======================================================
        if all_results:
            df = pd.DataFrame(all_results)
            csv_path = os.path.join(OUTPUT_DIR, "full_results.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"\n[SYSTEM] 💾 Results saved to {csv_path}")
        else:
            logger.warning("[SYSTEM] No results generated (All models failed?)")
            df = pd.DataFrame()
        
        return df
        
    def print_final_summary(self):
        print("\n\n")
        print("==================================================")
        print("               EXECUTION SUMMARY                  ")
        print("==================================================")
        
        summary_df = pd.DataFrame(self.execution_log)
        if not summary_df.empty:
            print(summary_df.to_markdown(index=False))
        else:
            print("No models were attempted.")

        failed_models = [x for x in self.execution_log if "FAILED" in x["Status"]]
        
        if failed_models:
            print("\n PIPELINE COMPLETED WITH ERRORS")
            print(f"   {len(failed_models)} process(es) failed to load or crashed.")
            print("   Check ./logs/execution.log for full tracebacks.")
            sys.exit(1)
        else:
            print("\n PIPELINE COMPLETED SUCCESSFULLY")

    def generate_report(self, df):
        if df.empty: return
        logger.info("\n[SYSTEM] 📊 Generating Comparative Analysis...")
        
        # Multidimensional Grouping for the Table
        asr_table = df.groupby(['Defense_Strategy', 'Model', 'Attack'])['Jailbroken'].mean() * 100
        print("\n--- Attack Success Rate (ASR) ---")
        print(asr_table)
        
        asr_table.to_csv(os.path.join(RESULT_DIR, "metrics_asr.csv"))
        self.generate_comparative_charts(df)


if __name__ == "__main__":
    pipeline = Pipeline()
    results_df = pipeline.run()
    
    if not results_df.empty:
        pipeline.generate_report(results_df)
    
    pipeline.print_final_summary()