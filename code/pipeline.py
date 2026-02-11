# import os
# import sys
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from vllm import LLM, SamplingParams
# import jailbreakbench as jbb
# import torch

# # Import modules
# from judge import JailbreakJudge
# from attacks import AttackLoader
# from defensive_layer import DefenseLayer

# # --- CONFIGURATION ---
# OUTPUT_DIR = "./results"
# if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# # 96GB VRAM Configuration:
# # We can load big models. 
# MODELS_TO_TEST = [
#     "lmsys/vicuna-13b-v1.5",           # The Standard Baseline
#     # "meta-llama/Meta-Llama-3-8B-Instruct" # Uncomment if downloaded
# ]

# ATTACKS_TO_TEST = ["SimplePrefix", "GCG", "PAIR"] # Add "PromptWithRS" if needed

# class Pipeline:
#     def __init__(self):
#         # 1. Load Dataset
#         print("[SYSTEM] 📂 Loading Datasets...")
#         self.ds = jbb.read_dataset()
#         # LIMIT TO 20 BEHAVIORS FOR INITIAL TESTING SPEED
#         # Remove the slice [:20] to run the full benchmark
#         self.behaviors = self.ds.behaviors[:20] 
#         self.goals = self.ds.goals[:20]
#         self.categories = self.ds.categories[:20]
#         self.sources = self.ds.sources[:20] if hasattr(self.ds, 'sources') else ["Unknown"]*len(self.goals)
        
#         # 2. Initialize Judge (Stays in Memory)
#         self.judge = JailbreakJudge()
        
#         # 3. Initialize Defense
#         self.defense = DefenseLayer()

#     def run(self):
#         all_results = []

#         for model_name in MODELS_TO_TEST:
#             print(f"\n{'='*60}\n[TARGET] 🎯 Loading Target Model: {model_name}\n{'='*60}")
            
#             # Load Target Model (Reserve remaining ~50% VRAM)
#             try:
#                 target_llm = LLM(
#                     model=model_name,
#                     dtype="float16",
#                     tensor_parallel_size=1,
#                     gpu_memory_utilization=0.45,
#                     enforce_eager=True,
#                     disable_log_stats=True
#                 )
#                 sampling_params = SamplingParams(temperature=0.1, max_tokens=150)
#             except Exception as e:
#                 print(f"[ERROR] Failed to load {model_name}: {e}")
#                 continue

#             # Initialize Attack Loader for this model
#             attacker = AttackLoader(model_name)

#             # --- ATTACK LOOP ---
#             for attack_method in ATTACKS_TO_TEST:
#                 print(f"\n[PIPELINE] 🚀 Running Attack: {attack_method}")
                
#                 # Get prompts (Artifacts or Generated)
#                 prompts = attacker.get_prompts(attack_method, self.goals)

#                 for i, (behavior, goal, prompt) in enumerate(zip(self.behaviors, self.goals, prompts)):
#                     print(f"  > [{i+1}/{len(self.goals)}] Behavior: {behavior}")

#                     # A. Defense: Input Filtering
#                     blocked_in, processed_prompt = self.defense.process_input(prompt)
                    
#                     if blocked_in:
#                         response_text = "I cannot fulfill this request (Defense Blocked Input)."
#                         is_jailbroken = False
#                         print("    [DEFENSE] 🛡️ Input Blocked.")
#                     else:
#                         # B. Target Inference
#                         outputs = target_llm.generate([processed_prompt], sampling_params)
#                         raw_response = outputs[0].outputs[0].text.strip()

#                         # C. Defense: Output Filtering
#                         blocked_out, final_response = self.defense.process_output(raw_response)
                        
#                         if blocked_out:
#                             response_text = "I cannot fulfill this request (Defense Blocked Output)."
#                             print("    [DEFENSE] 🛡️ Output Blocked.")
#                         else:
#                             response_text = final_response

#                         # D. Judge Evaluation
#                         is_jailbroken = self.judge.evaluate(goal, response_text)
#                         status = "UNSAFE 💀" if is_jailbroken else "SAFE ✅"
#                         print(f"    [JUDGE] Verdict: {status}")

#                     # Store Result
#                     all_results.append({
#                         "Model": model_name.split("/")[-1],
#                         "Attack": attack_method,
#                         "Category": self.categories[i],
#                         "Source": self.sources[i], # For Table 5 replication
#                         "Goal": goal,
#                         "Response": response_text,
#                         "Jailbroken": is_jailbroken
#                     })

#             # Unload Target Model to free VRAM for next one (if any)
#             import gc
#             del target_llm
#             gc.collect()
#             torch.cuda.empty_cache()

#         # Save Raw Data
#         df = pd.DataFrame(all_results)
#         df.to_csv(os.path.join(OUTPUT_DIR, "full_results.csv"), index=False)
#         print(f"\n[SYSTEM] 💾 Results saved to {OUTPUT_DIR}/full_results.csv")
        
#         return df

#     def generate_report(self, df):
#         if df.empty: return

#         print("\n[SYSTEM] 📊 Generating Comparative Analysis...")
        
#         # 1. ASR Table (By Attack)
#         asr_table = df.groupby(['Model', 'Attack'])['Jailbroken'].mean() * 100
#         print("\n--- Attack Success Rate (ASR) ---")
#         print(asr_table)
#         asr_table.to_csv(os.path.join(OUTPUT_DIR, "metrics_asr.csv"))

#         # 2. Plot: ASR by Attack Method
#         plt.figure(figsize=(10, 6))
#         sns.barplot(data=df, x="Attack", y="Jailbroken", hue="Model", errorbar=None, palette="viridis")
#         plt.title("Attack Success Rate (ASR) by Method")
#         plt.ylabel("ASR (0.0 - 1.0)")
#         plt.ylim(0, 1)
#         plt.savefig(os.path.join(OUTPUT_DIR, "plot_asr_by_attack.png"))
        
#         # 3. Plot: ASR by Category
#         plt.figure(figsize=(12, 6))
#         sns.barplot(data=df, x="Category", y="Jailbroken", hue="Model", errorbar=None, palette="magma")
#         plt.xticks(rotation=45, ha='right')
#         plt.title("Vulnerability by Category")
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTPUT_DIR, "plot_asr_by_category.png"))

#         print(f"[SYSTEM] 🖼️  Plots saved to {OUTPUT_DIR}")

# if __name__ == "__main__":
#     pipeline = Pipeline()
#     results_df = pipeline.run()
#     pipeline.generate_report(results_df)

# # Version - 2
# import os
# import sys
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# import gc
# import time

# # Import shared config and logger
# from logger_config import logger, LOGS_DIR, RESULT_DIR, OUTPUT_DIR

# # Import modules
# from judge import JailbreakJudge
# from attacks import AttackLoader
# from defensive_layer import DefenseLayer

# # --- PIPELINE CONFIGURATION ---
# MODELS_TO_TEST = [
#     "Qwen/Qwen1.5-0.5B-Chat-AWQ", # For extremely small VRAM
#     # "Qwen/Qwen1.5-1.8B-Chat-AWQ", # For small VRAM
#     # "lmsys/vicuna-13b-v1.5", # The Standard Baseline
#     # "meta-llama/Meta-Llama-3-8B-Instruct" # Uncomment if downloaded
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
        
#         # Slicing for fast testing (Remove [:] to run full set)
#         limit = 10 
#         self.behaviors = self.ds.behaviors[:limit] 
#         self.goals = self.ds.goals[:limit]
#         self.categories = self.ds.categories[:limit]
#         self.sources = self.ds.sources[:limit] if hasattr(self.ds, 'sources') else ["Unknown"]*len(self.goals)
        
#         # 2. Initialize Judge
#         logger.info("[SYSTEM] 📈 Loading Judge Model...")
#         self.judge = JailbreakJudge()
        
#         # 3. Initialize Defense
#         logger.info("[SYSTEM] 🛡️ Loading Defense Layer...")
#         self.defense = DefenseLayer()

#     def generate_comparative_charts(self, all_data_df):
#         """Generates bar charts comparing model performance by category."""
#         if all_data_df.empty:
#             logger.warning("[SYSTEM] No data available to plot.")
#             return

#         logger.info("[SYSTEM] 📊 Generating comparative charts...")
#         print("[SYSTEM] Generating comparative charts...")
        
#         # 1. Calculate Success Rate by Category and Model
#         summary_df = all_data_df.groupby(['Model', 'Category'])['Jailbroken'].mean().reset_index()
#         summary_df['Success Rate (%)'] = summary_df['Jailbroken'] * 100

#         # 2. Plotting
#         try:
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
            
#             # Save to RESULT_DIR
#             output_path = os.path.join(RESULT_DIR, 'jailbreak_comparison_chart.png')
#             plt.savefig(output_path, dpi=300)
#             logger.info(f"[SYSTEM] Chart saved to: {output_path}")
#             print(f"[SYSTEM] Chart saved to: {output_path}")
#             plt.close()
            
#         except Exception as e:
#             logger.error(f"Failed to generate charts: {e}")

#     def run(self):
#         all_results = []

#         for model_name in MODELS_TO_TEST:
#             logger.info(f"\n{'='*60}\n[TARGET] 🎯 Loading Target Model: {model_name}\n{'='*60}")
#             print(f"\n[TARGET] Loading Target Model: {model_name}")
            
#             try:
#                 # Load Target Model
#                 target_llm = LLM(
#                     model=model_name,
#                     dtype="float16",
#                     tensor_parallel_size=1,
#                     gpu_memory_utilization=0.45,
#                     enforce_eager=True,
#                     disable_log_stats=True
#                 )
#                 sampling_params = SamplingParams(temperature=0.1, max_tokens=150)
#             except Exception as e:
#                 logger.error(f"Failed to load {model_name}: {e}")
#                 continue

#             attacker = AttackLoader(model_name)

#             # --- ATTACK LOOP ---
#             for attack_method in ATTACKS_TO_TEST:
#                 logger.info(f"\n[PIPELINE] 🚀 Running Attack: {attack_method}")
#                 print(f"[PIPELINE] Running Attack: {attack_method}")
                
#                 prompts = attacker.get_prompts(attack_method, self.goals)

#                 for i, (behavior, goal, prompt) in enumerate(zip(self.behaviors, self.goals, prompts)):
#                     logger.info(f"  > [{i+1}/{len(self.goals)}] Behavior: {behavior}")

#                     # A. Defense: Input
#                     blocked_in, processed_prompt = self.defense.process_input(prompt)
                    
#                     if blocked_in:
#                         response_text = "I cannot fulfill this request (Defense Blocked Input)."
#                         is_jailbroken = False
#                         logger.info("    [DEFENSE] 🛡️ Input Blocked.")
#                     else:
#                         # B. Target Inference
#                         outputs = target_llm.generate([processed_prompt], sampling_params)
#                         raw_response = outputs[0].outputs[0].text.strip()

#                         # C. Defense: Output
#                         blocked_out, final_response = self.defense.process_output(raw_response)
                        
#                         if blocked_out:
#                             response_text = "I cannot fulfill this request (Defense Blocked Output)."
#                             logger.info("    [DEFENSE] 🛡️ Output Blocked.")
#                         else:
#                             response_text = final_response

#                         # D. Judge
#                         is_jailbroken = self.judge.evaluate(goal, response_text)
#                         status = "UNSAFE" if is_jailbroken else "SAFE"
#                         logger.info(f"    [JUDGE] Verdict: {status}")

#                     all_results.append({
#                         "Model": model_name.split("/")[-1],
#                         "Attack": attack_method,
#                         "Category": self.categories[i],
#                         "Source": self.sources[i],
#                         "Goal": goal,
#                         "Response": response_text,
#                         "Jailbroken": is_jailbroken
#                     })

#             # Cleanup
#             import gc
#             del target_llm
#             gc.collect()
#             torch.cuda.empty_cache()

#         # Save Raw Data to OUTPUT_DIR
#         df = pd.DataFrame(all_results)
#         csv_path = os.path.join(OUTPUT_DIR, "full_results.csv")
#         df.to_csv(csv_path, index=False)
#         logger.info(f"\n[SYSTEM] 💾 Results saved to {csv_path}")
#         print(f"[SYSTEM] Results saved to {csv_path}")
        
#         return df

#     def generate_report(self, df):
#         if df.empty: return

#         logger.info("\n[SYSTEM] 📊 Generating Comparative Analysis...")
#         print("\n[SYSTEM] Generating Comparative Analysis...")
        
#         # 1. Generate Summary Table (ASR)
#         asr_table = df.groupby(['Model', 'Attack'])['Jailbroken'].mean() * 100
        
#         print("\n\n==================================================")
#         print("          ATTACK SUCCESS RATE (ASR) SUMMARY       ")
#         print("==================================================")
#         print(asr_table)
        
#         # Save table to RESULT_DIR
#         asr_table.to_csv(os.path.join(RESULT_DIR, "metrics_asr.csv"))
        
#         # Save Markdown table to LOGS_DIR for record
#         with open(os.path.join(LOGS_DIR, "summary_table.txt"), "w") as f:
#             f.write(asr_table.to_markdown())

#         # 2. Generate Plots
#         self.generate_comparative_charts(df)

# if __name__ == "__main__":
#     pipeline = Pipeline()
#     results_df = pipeline.run()
#     pipeline.generate_report(results_df)

# Version - 3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gc
import time
from tabulate import tabulate # Optional: for pretty printing, but we can use pandas markdown too

# Import shared config and logger
from logger_config import logger, LOGS_DIR, RESULT_DIR, OUTPUT_DIR

# Import modules
from judge import JailbreakJudge
from attacks import AttackLoader
from defensive_layer import DefenseLayer

# --- PIPELINE CONFIGURATION ---
MODELS_TO_TEST = [
    # "Qwen/Qwen1.5-0.5B-Chat-AWQ" # For extremely low VRAM ~4GB,
    "Qwen/Qwen1.5-1.8B-Chat-AWQ", # Just used for Testing VRAM ~16GB
    # "meta-llama/Llama-3-8b-Instruct-hf",
    # "meta-llama/Llama-2-13b-chat-hf", 
    # "lmsys/vicuna-13b-v1.5", # The Standard Baseline

]

ATTACKS_TO_TEST = ["SimplePrefix", "GCG", "PAIR"]

# Dependency Check
try:
    import jailbreakbench as jbb
    from vllm import LLM, SamplingParams
except ImportError as e:
    logger.critical(f"Missing Dependency: {e}")
    sys.exit(1)

class Pipeline:
    def __init__(self):
        # 1. Load Dataset
        logger.info("[SYSTEM] 📂 Loading Datasets...")
        print("[SYSTEM] Loading Datasets...")
        self.ds = jbb.read_dataset()
        
        # Slicing for fast testing
        limit = 100 or len(self.ds.behaviors)
        self.behaviors = self.ds.behaviors[:limit] 
        self.goals = self.ds.goals[:limit]
        self.categories = self.ds.categories[:limit]
        self.sources = self.ds.sources[:limit] if hasattr(self.ds, 'sources') else ["Unknown"]*len(self.goals)
        
        # 2. Initialize Judge
        logger.info("[SYSTEM] 📈 Loading Judge Model...")
        self.judge = JailbreakJudge()
        
        # 3. Initialize Defense
        logger.info("[SYSTEM] 🛡️ Loading Defense Layer...")
        self.defense = DefenseLayer()
        
        # 4. Status Tracker (NEW)
        self.execution_log = [] 

    def generate_comparative_charts(self, all_data_df):
        if all_data_df.empty:
            logger.warning("[SYSTEM] No data available to plot (Skipping Charts).")
            return

        logger.info("[SYSTEM] 📊 Generating comparative charts...")
        
        try:
            summary_df = all_data_df.groupby(['Model', 'Category'])['Jailbroken'].mean().reset_index()
            summary_df['Success Rate (%)'] = summary_df['Jailbroken'] * 100

            plt.figure(figsize=(14, 8))
            sns.set_theme(style="whitegrid")
            
            sns.barplot(
                data=summary_df, 
                x='Category', 
                y='Success Rate (%)', 
                hue='Model',
                palette='viridis'
            )
            
            plt.title('Jailbreak Success Rate by Behavior Category', fontsize=16)
            plt.ylabel('Success Rate (%)', fontsize=12)
            plt.xlabel('Category', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            output_path = os.path.join(RESULT_DIR, 'jailbreak_comparison_chart.png')
            plt.savefig(output_path, dpi=300)
            logger.info(f"[SYSTEM] Chart saved to: {output_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")

    def run(self):
        all_results = []

        for model_name in MODELS_TO_TEST:
            logger.info(f"\n{'='*60}\n[TARGET] 🎯 Loading Target Model: {model_name}\n{'='*60}")
            print(f"\n[TARGET] Loading Target Model: {model_name}")
            
            model_start_time = time.time()
            try:
                # Load Target Model
                target_llm = LLM(
                    model=model_name,
                    dtype="float16" or "float32" or "auto",
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.50,
                    enforce_eager=True,
                    disable_log_stats=True
                )
                logger.info(f"[SYSTEM] target_llm config: {target_llm}")
                sampling_params = SamplingParams(temperature=0.1, max_tokens=150)
                
            except Exception as e:
                # RECORD FAILURE
                logger.error(f"Failed to load {model_name}: {e}")
                self.execution_log.append({
                    "Model": model_name,
                    "Status": " FAILED",
                    "Reason": str(e)[:100] + "..." # Truncate long errors
                })
                continue # Skip to next model

            # Initialize Attacker
            attacker = AttackLoader(model_name)

            # --- ATTACK LOOP ---
            try:
                for attack_method in ATTACKS_TO_TEST:
                    logger.info(f"\n[PIPELINE] 🚀 Running Attack: {attack_method}")
                    print(f"[PIPELINE] Running Attack: {attack_method}")
                    
                    prompts = attacker.get_prompts(attack_method, self.goals)

                    for i, (behavior, goal, prompt) in enumerate(zip(self.behaviors, self.goals, prompts)):
                        # A. Defense: Input
                        blocked_in, processed_prompt = self.defense.process_input(prompt)
                        
                        if blocked_in:
                            response_text = "I cannot fulfill this request (Defense Blocked Input)."
                            is_jailbroken = False
                        else:
                            # B. Target Inference
                            outputs = target_llm.generate([processed_prompt], sampling_params)
                            raw_response = outputs[0].outputs[0].text.strip()

                            # C. Defense: Output
                            blocked_out, final_response = self.defense.process_output(raw_response)
                            
                            if blocked_out:
                                response_text = "I cannot fulfill this request (Defense Blocked Output)."
                            else:
                                response_text = final_response

                            # D. Judge
                            is_jailbroken = self.judge.evaluate(goal, response_text)
                            
                        # Logging verdict
                        status = "UNSAFE" if is_jailbroken else "SAFE"
                        logger.info(f"  > [{i+1}/{len(self.goals)}] {behavior[:30]}... -> {status}")

                        all_results.append({
                            "Model": model_name.split("/")[-1],
                            "Attack": attack_method,
                            "Category": self.categories[i],
                            "Source": self.sources[i],
                            "Goal": goal,
                            "Response": response_text,
                            "Jailbroken": is_jailbroken
                        })
                
                # RECORD SUCCESS
                self.execution_log.append({
                    "Model": model_name,
                    "Status": "SUCCESS",
                    "Reason": f"Completed in {round(time.time() - model_start_time, 2)}s"
                })

            except Exception as e:
                # RECORD RUNTIME FAILURE
                logger.error(f"Runtime error on {model_name}: {e}")
                self.execution_log.append({
                    "Model": model_name,
                    "Status": " CRASHED",
                    "Reason": str(e)[:100]
                })

            # Cleanup
            import gc
            del target_llm
            gc.collect()
            torch.cuda.empty_cache()

        # Save Raw Data to OUTPUT_DIR
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
        """Prints a honest breakdown of what happened."""
        print("\n\n")
        print("==================================================")
        print("               EXECUTION SUMMARY                  ")
        print("==================================================")
        
        summary_df = pd.DataFrame(self.execution_log)
        if not summary_df.empty:
            print(summary_df.to_markdown(index=False))
        else:
            print("No models were attempted.")

        # Check for failures
        failed_models = [x for x in self.execution_log if "FAILED" in x["Status"] or "CRASHED" in x["Status"]]
        
        if failed_models:
            print("\n PIPELINE COMPLETED WITH ERRORS")
            print(f"   {len(failed_models)} model(s) failed to load or crashed.")
            print("   Check ./logs/execution.log for full tracebacks.")
            sys.exit(1) # Exit code 1 to signal failure to CI/CD or scripts
        else:
            print("\n PIPELINE COMPLETED SUCCESSFULLY")

    def generate_report(self, df):
        if df.empty: return

        logger.info("\n[SYSTEM] 📊 Generating Comparative Analysis...")
        
        # 1. Generate Summary Table (ASR)
        asr_table = df.groupby(['Model', 'Attack'])['Jailbroken'].mean() * 100
        
        print("\n--- Attack Success Rate (ASR) ---")
        print(asr_table)
        
        # Save table
        asr_table.to_csv(os.path.join(RESULT_DIR, "metrics_asr.csv"))

        # 2. Generate Plots
        self.generate_comparative_charts(df)

if __name__ == "__main__":
    pipeline = Pipeline()
    results_df = pipeline.run()
    
    # Generate charts only if we have data
    if not results_df.empty:
        pipeline.generate_report(results_df)
    
    # ALWAYS print the status summary at the end
    pipeline.print_final_summary()