# ./model_loader.py

import os
import re
import gc
import psutil
import torch
import traceback
from typing import Tuple, Optional, Dict, Any

from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError, HfHubHTTPError

from logger_config import logger

def get_utilization(model_id: str) -> float:
    model_id = model_id.lower()
    if "gemma" in model_id:
        return 0.70  # Heavy vocab overhead
    if "qwen" in model_id:
        return 0.77  # Moderate vocab overhead
    return 0.80      # Standard Llama-based models

class HardwareAwareModelLoader:
    """
    Centralized Hardware Manager. Dynamically assesses VRAM, System RAM, and GPU architecture 
    to select the safest and most optimal model format (AWQ vs FP16) and parses composite model strings.
    """

    @staticmethod
    def parse_model_string(raw_string: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Universal string parser handling base_repo fallbacks (|)."""
        base_repo = None
        if "|" in raw_string:
            raw_string, base_repo = raw_string.split("|", 1)
            
        repo_id = raw_string
        gguf_file = None
        if raw_string.lower().endswith(".gguf") and not os.path.exists(raw_string):
            parts = raw_string.split("/")
            if len(parts) >= 3:
                repo_id = "/".join(parts[:2])
                gguf_file = "/".join(parts[2:])
                
        return repo_id, gguf_file, base_repo

    @staticmethod
    def log_system_memory(phase: str):
        """Logs exact System RAM and GPU VRAM usage for absolute memory visibility."""
        try:
            # System RAM
            mem = psutil.virtual_memory()
            total_ram = mem.total / (1024 ** 3)
            used_ram = mem.used / (1024 ** 3)
            free_ram = mem.available / (1024 ** 3)
            
            # GPU VRAM Memory Usage
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                allocated_vram = torch.cuda.memory_allocated(device) / (1024**3)
                reserved_vram = torch.cuda.memory_reserved(device) / (1024**3)
                free_vram = total_vram - reserved_vram
                
                logger.info(
                    f"[MEMORY] 📊 {phase} | "
                    f"RAM: {used_ram:.1f}GB used / {free_ram:.1f}GB free (Total RAM: {total_ram:.1f}GB) | "
                    f"GPU VRAM: {allocated_vram:.1f}GB allocated ({reserved_vram:.1f}GB reserved) / {free_vram:.1f}GB free"
                )
            else:
                logger.info(f"[MEMORY] 📊 {phase} | RAM: {used_ram:.1f}GB used / {free_ram:.1f}GB free (Total RAM: {total_ram:.1f}GB) | GPU: Not Available")
                
        except Exception as e:
            logger.warning(f"[MEMORY] ⚠️ Could not poll system memory: {e}")

    @staticmethod
    def _verify_huggingface_access(model_id: str):
        """Performs a pre-flight check to verify HF repository status, access, and cache location."""
        
        # 1. Dynamically parse the composite string
        repo_id, gguf_file, base_repo = HardwareAwareModelLoader.parse_model_string(model_id)
        target_repo = base_repo if base_repo else repo_id
        
        logger.info(f"\n[HARDWARE] 🔍 Diagnosing Hugging Face Repository/ Path: '{target_repo}'")

        # Bypass network check if the user passed a local GGUF file path
        if target_repo.lower().endswith('.gguf') and os.path.exists(target_repo):
            logger.info(f"[HARDWARE] ✅ Local GGUF file detected. Bypassing Hugging Face API check.")
            return

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("[HARDWARE] ⚠️ No HF_TOKEN found in environment. Gated models will fail.")

        try:
            # 1. Ping Hugging Face API
            info = model_info(target_repo, token=hf_token)
            visibility = "Private" if info.private else "Public"
            logger.info(f"[HARDWARE] ✅ Repository found. Access Granted. (Visibility: {visibility})")
            
            # 2. Check local HDD cache path
            # Hugging Face caches models in ~/.cache/huggingface/hub/models--<namespace>--<model_name>
            cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
            safe_model_dir = f"models--{target_repo.replace('/', '--')}"
            full_cache_path = os.path.join(cache_dir, safe_model_dir)
            
            if os.path.exists(full_cache_path):
                logger.info(f"[HARDWARE] 📂 Model weights are already downloaded in HDD: {full_cache_path}")
            else:
                logger.info(f"[HARDWARE] 🌐 Model weights not found locally. vLLM will download them to: {cache_dir}")
                
        except RepositoryNotFoundError:
            logger.critical(f"[HARDWARE] ❌ The repository '{target_repo}' does not exist or was deleted.")
            raise
        except GatedRepoError:
            logger.critical(f"[HARDWARE] ❌ '{target_repo}' is gated. You must accept the terms on the Hugging Face website.")
            raise
        except HfHubHTTPError as e:
            logger.critical(f"[HARDWARE] ❌ Hugging Face API Network Error: {e}")
            raise
        except Exception as e:
            logger.warning(f"[HARDWARE] ⚠️ Unexpected error during HF diagnostic: {e}")

    @staticmethod
    def get_hardware_optimized_model(base_model_id: str) -> Tuple[str, Optional[str], int]:
        logger.info("[HARDWARE] 🟢 Entering TRY block: Hardware capability assessment")
        try:
            # Dynamically extract base repo to avoid 3-part GGUF string errors in vLLM
            repo_id, gguf_file, base_repo = HardwareAwareModelLoader.parse_model_string(base_model_id)
            target_repo = base_repo if base_repo else repo_id
            
            # Pre-flight checks
            HardwareAwareModelLoader.log_system_memory("Pre-Model Allocation")
            HardwareAwareModelLoader._verify_huggingface_access(base_model_id)

            # Cloud/Local GPU overrides
            num_gpus = int(os.environ.get("VLLM_TARGET_GPUS", 1))
            
            if num_gpus == 0 or not torch.cuda.is_available():
                logger.warning("[HARDWARE] ⚠️ No GPUs detected. Forcing CPU mode.")
                return target_repo, None, 0
            
            try:
                compute_cap = torch.cuda.get_device_capability(0)

                # Check the infrastructure is A100 or V100
                is_a100_or_newer = (compute_cap[0] >= 8)

                logger.debug(f"[HARDWARE] Detected CUDA Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

            except Exception as e:
                logger.error(f"[HARDWARE] 🚨 CUDA capability check failed: {e}. Enforcing safe Pascal/Volta defaults.")
                compute_cap = (7, 0)
                is_a100_or_newer = False
            
            base_lower = target_repo.lower()
            
            # Check if the model is natively a GGUF file (no base repo fallback provided)
            if "gguf" in base_lower or base_lower.endswith(".gguf") or gguf_file:
                if not base_repo:
                    logger.info("[HARDWARE] 📦 Native GGUF format detected! Routing to vLLM's native GGUF engine.")
                    logger.info("[HARDWARE] ⚡ Configuring vLLM to map GGUF directly to GPU VRAM for maximum speed.")
                    model = target_repo
                    quant = "gguf" # Instructs vLLM to use its native GGUF engine
                    return model, quant, num_gpus

            if is_a100_or_newer:
                # A100 / Hopper LOGIC
                if "awq" in base_lower:
                    logger.info("[HARDWARE] 🚀 Modern GPU detected. Target is natively AWQ.")
                    model = target_repo
                    quant = "awq"
                # elif any(x in base_lower for x in ["vicuna-13b", "llama-2-70b", "llama-3.1"]):
                #     logger.info(f"[HARDWARE] 🚀 Modern GPU detected. Upgrading {target_repo} to AWQ for speed.")
                #     model = f"{target_repo}-AWQ" 
                #     quant = "awq"
                else:
                    logger.info("[HARDWARE] 🚀 Modern GPU detected. Loading standard FP16 model.")
                    model = target_repo
                    quant = None
            else:
                # V100 / Pascal / Turing LOGIC
                logger.info(f"[HARDWARE] ⚠️ Compute {compute_cap[0]}.{compute_cap[1]} detected. Enforcing stable FP16.")
                quant = None
                
                # Safely strip AWQ suffix for V100 to force FP16 fallback
                if "awq" in base_lower:
                    model = re.sub(r'(?i)-awq', '', target_repo)
                    logger.warning(f"[HARDWARE] AWQ is unstable on this architecture. Reverted to FP16 repository: {model}")
                else:
                    model = target_repo
                    
            return model, quant, num_gpus

        except Exception as e:
            logger.error(f"[HARDWARE] ❌ EXCEPTION in hardware optimization: {e}")
            logger.debug(traceback.format_exc())
            raise e
        finally:
            logger.info("[HARDWARE] 🏁 Exiting FINALLY block: Hardware assessment complete")

    @staticmethod
    def get_optimal_target_config(preferred_base_model: str) -> Dict[str, Any]:
        """
        Generates the complete vLLM initialization dictionary based on strict hardware constraints.
        """
        logger.info("\n" + "="*60)
        logger.info(f"[HARDWARE] 🟢 Entering TRY block: Generating vLLM Engine Configuration")
        try:
            final_model, quant, num_gpus = HardwareAwareModelLoader.get_hardware_optimized_model(preferred_base_model)
            
            config = {
                "model": final_model,
                "dtype": "auto" if quant == "gguf" else "float16",
                "quantization": quant,
                "gpu_memory_utilization": get_utilization(final_model), 
                "max_model_len": 4096,     # Strict limit to protect VRAM from KV Cache explosions
                # "max_model_len": 2048,
                "enforce_eager": True,     # Prevents CUDA graph compilation timeouts
                "disable_log_stats": True,
                "trust_remote_code": True,
                "disable_custom_all_reduce": True, # Force the engine to use standard Torch RMSNorm
                # "tensor_parallel_size": 1, # Explicitly locking to 1 to bypass Gloo/NCCL crashes on V100s
                "tensor_parallel_size": 2, # Tells vLLM to shard the model across both 32GB V100 GPUs
                "max_num_seqs": 4, # Forces vLLM to only process 4 prompts at a time instead of 10+
                "distributed_executor_backend": "mp", # Force Multi-Processing instead of Ray
            }
            
            # Eradicate the distributed backend key if single GPU to prevent network boot
            if "distributed_executor_backend" in config:
                del config["distributed_executor_backend"]

            logger.info(
                f"[HARDWARE] 🏁 Final vLLM Engine Payload -> "
                f"Model: '{final_model}' | Quant: {quant} | TP_Size: {config['tensor_parallel_size']} | Max_Len: {config['max_model_len']}"
            )
            return config

        except Exception as e:
            logger.error(f"[HARDWARE] ❌ EXCEPTION generating vLLM config: {e}")
            logger.debug(traceback.format_exc())
            raise e
        finally:
            logger.info("[HARDWARE] 🏁 Exiting FINALLY block: vLLM Config Factory")
            
    @staticmethod
    def force_vram_purge():
        """
        Utility to be called by pipeline.py after evaluation completes 
        to mathematically guarantee a pristine GPU state.
        """
        logger.info("[HARDWARE] 🟢 Entering TRY block: Forcing manual OS-Level Memory Purge")
        try:
            HardwareAwareModelLoader.log_system_memory("Pre-Purge")
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect() # Clears shared memory segments
            HardwareAwareModelLoader.log_system_memory("Post-Purge")
        except Exception as e:
            logger.error(f"[HARDWARE] ❌ EXCEPTION during VRAM purge: {e}")
        finally:
            logger.info("[HARDWARE] 🏁 Exiting FINALLY block: Purge sequence complete")