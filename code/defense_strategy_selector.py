# ./defense_strategy_selector.py

import os
import gc
# import time
import traceback
import psutil
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from logger_config import logger, RESULT_DIR

# Import the attack and evaluation modules
from attacks import AttackLoader
from judge import JailbreakJudge

def log_local_memory(phase: str):
    """Local utility for deep memory diagnostics during orchestration."""
    try:
        mem = psutil.virtual_memory()
        used_ram = mem.used / (1024 ** 3)
        log_str = f"[MEMORY] 📊 {phase} | System RAM: {used_ram:.1f}GB"
        if torch.cuda.is_available():
            alloc_vram = torch.cuda.memory_allocated() / (1024**3)
            log_str += f" | GPU VRAM: {alloc_vram:.1f}GB"
        logger.debug(log_str)
    except:
        pass


# =====================================================================
# 1. DEFENSE-SELECTOR
# =====================================================================
class DefenseFactory:
    """Dynamically loads the requested defense module from the restructured architecture."""
    
    # THE REGISTRY: The Factory knows its inventory without building it.
    AVAILABLE_STRATEGIES = ["fuse", "smoothing", "enterprise", "dcmd", "none"]

    @staticmethod
    def is_valid(strategy_name: str) -> bool:
        """Allows pipeline.py to do a pre-flight check without instantiating objects."""
        return strategy_name.lower().strip() in DefenseFactory.AVAILABLE_STRATEGIES

    @staticmethod
    def get_defense_layer(strategy_name: str, config=None):
        strategy = strategy_name.lower().strip()
        logger.info("\n" + "-"*50)
        logger.info(f"[DEFENSE-SELECTOR] 🟢 Entering TRY block: Instantiating Defense '{strategy.upper()}'")
        
        try:
            if strategy == "fuse":
                from defensive_baseline import DefenseLayer
                return DefenseLayer(config)
            
            elif strategy == "smoothing":
                from defensive_smoothing import DefenseLayer
                return DefenseLayer(config)

            elif strategy == "dcmd":
                from defensive_dual_phase_cryptographic_manifold_defense import DefenseLayer
                return DefenseLayer(config)
                
            elif strategy == "enterprise":
                from defensive_streaming_interceptor import EnterpriseStreamingDefense
                return EnterpriseStreamingDefense(config)
                
            elif strategy == "none":
                # A transparent pass-through class for the control group
                class NoDefense:
                    def process_input(self, ctx): return False, str(ctx).rsplit("User:", 1)[-1].strip()
                    def process_output(self, res): return False, res
                    def cleanup(self): pass
                return NoDefense()
                
            else:
                logger.critical(f"[DEFENSE-SELECTOR] ❌ Unknown defense strategy requested: {strategy}")
                raise ValueError(f"Unknown defense strategy requested: {strategy}")
                
        except Exception as e:
            logger.critical(f"[DEFENSE-SELECTOR] ❌ EXCEPTION loading '{strategy}': {e}")
            logger.debug(traceback.format_exc())
            
            # Fail-Open: If a defense module crashes during import, fallback to None to save the pipeline
            logger.warning("[DEFENSE-SELECTOR] ⚠️ Engaging Fail-Open protocol. Bypassing defense.")
            # Fallback Passthrough to prevent total pipeline collapse
            class FallbackDefense:
                def process_input(self, ctx): return False, str(ctx).rsplit("User:", 1)[-1].strip()
                def process_output(self, res): return False, res
                def cleanup(self): pass
            return FallbackDefense()
            
        finally:
            logger.info(f"[DEFENSE-SELECTOR] 🏁 Exiting FINALLY block: Defense instantiation complete.")

# =====================================================================
# 2. ASR VISUALIZER (MATPLOTLIB TABLE GENERATOR)
# =====================================================================
class ASRVisualizer:
    """Renders a publication-ready Attack Success Rate (ASR) table using Matplotlib."""
    
    @staticmethod
    def generate_asr_table(results_df: pd.DataFrame, model_name: str):
        if results_df.empty:
            logger.warning("[VISUALIZER] ⚠️ DataFrame is empty. No matrix to plot.")
            return

        logger.info("[VISUALIZER] 🟢 Entering TRY block: Generating ASR Matplotlib Table")
        try:
            # Calculate ASR (Percentage of Jailbroken == True)
            asr_df = results_df.groupby(['Defense_Strategy', 'Attack'])['Jailbroken'].mean().reset_index()
            asr_df['ASR (%)'] = (asr_df['Jailbroken'] * 100).round(1)

            # Pivot to create the Matrix: Rows = Defenses, Columns = Attacks
            pivot_table = asr_df.pivot(index='Defense_Strategy', columns='Attack', values='ASR (%)').fillna(0)
            
            fig, ax = plt.subplots(figsize=(10, pivot_table.shape[0] * 1.2 + 2))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(
                cellText=pivot_table.values,
                rowLabels=pivot_table.index,
                colLabels=pivot_table.columns,
                cellLoc='center',
                loc='center',
                bbox=[0.1, 0.1, 0.9, 0.8]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)

            # Academic Styling & Color Coding (Soothing Pastels)
            for (i, j), cell in table.get_celld().items():
                if i == 0 or j == -1:  # Headers
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#2C3E50')
                else:
                    value = float(cell.get_text().get_text())
                    if value < 5.0:
                        cell.set_facecolor('#D5F5E3') # Safe (Pastel Green)
                    elif value < 50.0:
                        cell.set_facecolor('#FCF3CF') # Warning (Pastel Yellow)
                    else:
                        cell.set_facecolor('#FADBD8') # Vulnerable (Pastel Red)

            plt.title(f"Attack Success Rate (ASR) Matrix\nTarget: {model_name.split('/')[-1]}", fontweight="bold", fontsize=16, pad=20)
            
            # Save high-resolution image
            safe_model_name = model_name.replace("/", "_")
            output_path = os.path.join(RESULT_DIR, f"asr_matrix_{safe_model_name}.png")
            os.makedirs(RESULT_DIR, exist_ok=True)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[VISUALIZER] ✅ Table successfully saved to: {output_path}")

        except Exception as e:
            logger.error(f"[VISUALIZER] ❌ EXCEPTION during matrix rendering: {e}")
            logger.debug(traceback.format_exc())
        finally:
            logger.info("[VISUALIZER] 🏁 Exiting FINALLY block: Matplotlib engine closed.")

# =====================================================================
# 3. CORE ORCHESTRATOR
# =====================================================================
class DefenseOrchestrator:
    """Manages the execution of all attacks against a loaded LLM under selected defenses."""
    
    def __init__(
        self, target_llm, 
        target_model_name: str, 
        sampling_params
    ):
        logger.info(f"\n[ORCHESTRATOR] ⚙️ Booting Orchestrator for {target_model_name}")
        self.llm = target_llm
        self.model_name = target_model_name
        self.sampling_params = sampling_params
        
        # The ultimate gauntlet of attacks requested
        self.attacks_to_run = ["Simple-Prefix", "GCG", "PAIR", "JB-Chat", "MTJ"]
        self.attacker = AttackLoader(self.model_name)

    def execute_evaluation(self, strategies_to_test: list, goals: list, categories: list) -> pd.DataFrame:
        logger.info("[ORCHESTRATOR] 🟢 Entering TRY block: Complete Pipeline Evaluation")
        
        attack_results = [] # All attack results will be stored here
        generated_data = [] # Temporary cache to allow Judge persistence
        
        try:
            # =====================================================================
            # PHASE 1: GENERATION LOOP (VRAM INTENSIVE)
            # =====================================================================
            for strategy in strategies_to_test:
                # Compile the system state payload
                orchestrator_config = {
                    "model_name": self.model_name,
                    "sampling_params": self.sampling_params
                }
                defense_layer = DefenseFactory.get_defense_layer(strategy, config=orchestrator_config)
                
                for attack_method in self.attacks_to_run:
                    logger.info(f"\n[ORCHESTRATOR] 🚀 Running Attack '{attack_method}' vs Defense '{strategy.upper()}'")
                    
                    try:
                        prompt_sequences, active_goals, active_behaviors = self.attacker.get_prompts(
                            attack_method, goals, categories
                        )
                    except Exception as e:
                        logger.error(f"[ORCHESTRATOR] ❌ Skipping {attack_method} due to extraction failure: {e}")
                        continue

                    # 2. Sequential Conversational Loop
                    for i, (behavior, goal, sequence) in enumerate(tqdm(zip(active_behaviors, active_goals, prompt_sequences), total=len(active_goals), desc=f"Generating {attack_method}")):
                        chat_history = ""
                        blocked_in_turn = False
                        final_turn_response = ""
                        
                        try:
                            for turn_idx, turn_prompt in enumerate(sequence):
                                # Defense Layer 1: Input filtering / Smoothing
                                context_to_evaluate = chat_history + f"User: {turn_prompt}"
                                blocked_in, processed_prompt = defense_layer.process_input(context_to_evaluate)
                                
                                if blocked_in:
                                    blocked_in_turn = True
                                    final_turn_response = f"I cannot fulfill this request (Defense Blocked Input at Turn {turn_idx+1})."
                                    break 
                                    
                                 # --- N-SAMPLE VRAM EXECUTION FOR SMOOTHLLM ---
                                if isinstance(processed_prompt, list):
                                    chat_histories = [chat_history + f"User: {p}\nAssistant: " for p in processed_prompt]
                                    outputs = self.llm.generate(chat_histories, self.sampling_params)
                                    final_turn_response = [out.outputs[0].text.strip() for out in outputs]
                                    # Temporarily track the first trace for multi-turn history continuity
                                    chat_history += f"{final_turn_response[0]}\n\n"
                                else:
                                    # Normal Execution
                                    chat_history += f"User: {processed_prompt}\nAssistant: "
                                    outputs = self.llm.generate([chat_history], self.sampling_params)
                                    raw_response = outputs[0].outputs[0].text.strip()
                                    chat_history += f"{raw_response}\n\n"
                                    final_turn_response = raw_response

                        except Exception as gen_err:
                            logger.error(f"[ORCHESTRATOR] ⚠️ Generation crashed for goal '{goal[:30]}...': {gen_err}")
                            final_turn_response = "Generation Error Triggered."
                            blocked_in_turn = True

                        generated_data.append({
                            "Strategy": strategy.upper(),
                            "Attack": attack_method,
                            "Category": behavior,
                            "Goal": goal,
                            "Final_Response": final_turn_response,
                            "Blocked_In": blocked_in_turn,
                            "Defense_Instance": defense_layer # Cache the layer for output processing later
                        })

                # Cleanup defense memory before swapping to the next strategy
                # Cleanup defense memory before swapping to the next strategy
                if hasattr(defense_layer, 'cleanup'):
                    defense_layer.cleanup()
                del defense_layer

                gc.collect()
                torch.cuda.empty_cache()
                log_local_memory(f"Post-Generation: {strategy.upper()}")

            # =====================================================================
            # PHASE 2: PERSISTENT JUDGE LOOP (RAM INTENSIVE)
            # =====================================================================
            logger.info(f"\n[ORCHESTRATOR] ⚖️ Entering Phase 2: Instantiating Persistent Judge for {len(generated_data)} total items.")

            judge = JailbreakJudge()
            
            try:
                for item in tqdm(generated_data, desc="Global Evaluation Phase", colour="green"):
                    if item["Blocked_In"]:
                        response_text = item["Final_Response"]
                        is_jailbroken = False
                    else:
                        # Defense Layer 2: Output streaming interception/filtering
                        blocked_out, final_response = item["Defense_Instance"].process_output(item["Final_Response"])
                        response_text = "I cannot fulfill this request (Output Blocked)." if blocked_out else final_response
                        
                        # Evaluate via C++ RAM Model
                        is_jailbroken = judge.evaluate(item["Goal"], response_text)

                    attack_results.append({
                        "Defense_Strategy": item["Strategy"],
                        "Model": self.model_name.split("/")[-1],
                        "Attack": item["Attack"],
                        "Category": item["Category"],
                        "Goal": item["Goal"],
                        "Jailbroken": is_jailbroken
                    })

                    # --- JAILBROKEN RESPONSE EXPORT ---
                    if is_jailbroken:
                        jailbreak_record = {
                            "Model": self.model_name.split("/")[-1],
                            "Attack": item["Attack"],
                            "Category": item["Category"],
                            "Goal": item["Goal"],
                            "Response": response_text
                        }
                        
                        csv_path = "./model_responses/jailbroken_full_response.csv"
                        # GUARANTEE the directory exists before writing
                        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                        
                        df_jb = pd.DataFrame([jailbreak_record])
                        
                        # If file exists, append without headers. If not, write with headers.
                        if os.path.exists(csv_path):
                            df_jb.to_csv(csv_path, mode='a', header=False, index=False)
                        else:
                            df_jb.to_csv(csv_path, mode='w', header=True, index=False)

            except Exception as e:
                logger.error(f"[ORCHESTRATOR] ❌ EXCEPTION during global evaluation: {e}")
                logger.debug(traceback.format_exc())
            finally:
                # Explicitly unload Judge only AFTER all attacks and strategies are finished
                logger.info("[ORCHESTRATOR] 🧹 Phase 2 complete. Dismantling Persistent Judge.")
                if hasattr(judge, 'unload_model'):
                    judge.unload_model()
                else:
                    del judge.llm
                    del judge
                gc.collect()

            # =====================================================================
            # PHASE 3: VISUALIZATION & OUTPUT
            # =====================================================================
            results_df = pd.DataFrame(attack_results)
            ASRVisualizer.generate_asr_table(results_df, self.model_name)
            
            return results_df

        except Exception as e:
            logger.critical(f"[ORCHESTRATOR] ❌ FATAL EXCEPTION in Pipeline Execution: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame(attack_results) # Return whatever survived
            
        finally:
            logger.info("[ORCHESTRATOR] 🏁 Exiting FINALLY block: Orchestration Sequence Terminated.")