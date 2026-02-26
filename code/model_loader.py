import re
import torch
from logger_config import logger

class HardwareAwareModelLoader:
    """
    Centralized Hardware Manager. Dynamically assesses VRAM and GPU architecture 
    to select the safest and most optimal model format (AWQ vs FP16).
    """

    @staticmethod
    def get_hardware_optimized_model(base_model_id):
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            return base_model_id, None, 0
        
        compute_cap = torch.cuda.get_device_capability(0)
        is_a100_or_newer = (compute_cap[0] >= 8)
        
        if is_a100_or_newer:
            # A100 LOGIC
            # Map standard model names to known AWQ HuggingFace repos ONLY if we know they exist
            if "vicuna-13b-v1.5" in base_model_id.lower() and "awq" not in base_model_id.lower():
                logger.info(f"[HARDWARE] 🚀 A100 detected. Upgrading Vicuna to AWQ.")
                model = "TheBloke/Vicuna-13B-v1.5-AWQ"
                quant = "awq"
            elif "llama-2-70b" in base_model_id.lower() and "awq" not in base_model_id.lower():
                logger.info(f"[HARDWARE] 🚀 A100 detected. Upgrading Llama-70B to AWQ.")
                model = "TheBloke/Llama-2-70B-Chat-AWQ"
                quant = "awq"
            elif "awq" in base_model_id.lower():
                logger.info(f"[HARDWARE] 🚀 A100 detected. Loading requested AWQ model.")
                model = base_model_id
                quant = "awq"
            else:
                # Do NOT blindly append -AWQ to unknown repos. Leave them as FP16.
                logger.info(f"[HARDWARE] 🚀 A100 detected. Loading standard FP16 model.")
                model = base_model_id
                quant = None
        else:
            # V100 LOGIC
            logger.info(f"[HARDWARE] ⚠️ Compute {compute_cap[0]}.{compute_cap[1]} detected. Enforcing standard FP16.")
            quant = None
            
            # Revert AWQ repos to official standard repos if V100
            if "vicuna" in base_model_id.lower() and "awq" in base_model_id.lower():
                model = "lmsys/vicuna-13b-v1.5"
            elif "llama-2-70b" in base_model_id.lower() and "awq" in base_model_id.lower():
                model = "meta-llama/Llama-2-70b-chat-hf"
            else:
                # Safely strip AWQ suffix for unknown models so they don't crash
                model = re.sub(r'(?i)-awq', '', base_model_id)
                
        return model, quant, num_gpus

    @staticmethod
    def get_optimal_target_config(preferred_base_model):
        """
        Used by pipeline.py to get the complete vLLM dictionary for Target Models.
        """
        final_model, quant, num_gpus = HardwareAwareModelLoader.get_hardware_optimized_model(preferred_base_model)
        
        config = {
            "model": final_model,
            "dtype": "float16",
            "quantization": quant,
            "tensor_parallel_size": num_gpus,
            "gpu_memory_utilization": 0.85, # Target inference baseline
            # "distributed_executor_backend": "mp",
            "max_model_len": 2048,
            "enforce_eager": True,
            "disable_log_stats": True,
            "trust_remote_code": True
        }
        logger.info(f"[HARDWARE] 🏁 Target Config -> Model: {final_model} | Quant: {quant} | GPUs: {num_gpus}")
        return config