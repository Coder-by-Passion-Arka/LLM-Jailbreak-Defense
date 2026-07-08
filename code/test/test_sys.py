# ./test_sys.py
import os
import sys
import time
import shutil
import psutil
import torch
import torch.distributed as dist
import gc
from logger_config import logger

def test_distributed_env():
    logger.info("="*60)
    logger.info("[SYS-TEST] 🔍 Initiating HYPER-PARANOID Multi-GPU Pre-Flight Diagnostic...")
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        logger.error("[SYS-TEST] ❌ Multi-GPU test failed: Less than 2 GPUs detected.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 1. TEST SYSTEM RAM & VRAM CAPACITIES
    # ---------------------------------------------------------
    try:
        logger.info("[SYS-TEST] 🧠 Checking System RAM and VRAM Capacities...")
        mem = psutil.virtual_memory()
        free_ram_gb = mem.available / (1024**3)
        total_ram_gb = mem.total / (1024**3)
        logger.info(f"[SYS-TEST] 💻 System RAM: {free_ram_gb:.2f} GB Available / {total_ram_gb:.2f} GB Total")
        
        if free_ram_gb < 16.0:
            logger.warning("[SYS-TEST] ⚠️ System RAM is low (< 16GB). Moving models to/from VRAM may cause swapping and slowdowns.")

        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            name = torch.cuda.get_device_name(i)
            total_vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            # Calculate truly free VRAM by subtracting what is already reserved by PyTorch or ghost processes
            free_vram = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)) / (1024**3)
            
            logger.info(f"[SYS-TEST] 🎮 GPU {i}: {name} | Compute {cap[0]}.{cap[1]} | VRAM: {free_vram:.2f} GB Free / {total_vram:.2f} GB Total")
            
            if cap[0] < 8:
                logger.warning(f"[SYS-TEST] ⚠️ GPU {i} does not support Bfloat16. dtype='half' is REQUIRED.")
    except Exception as e:
        logger.error(f"[SYS-TEST] ❌ Failed to probe system memory: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. TEST HARD DRIVE SPACE (HUGGING FACE CACHE)
    # ---------------------------------------------------------
    try:
        logger.info("[SYS-TEST] 💾 Checking Storage Space for Model Weights...")
        hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        
        # Ensure the directory exists before checking
        os.makedirs(hf_cache, exist_ok=True)
        total, used, free = shutil.disk_usage(hf_cache)
        free_gb = free / (1024**3)
        
        logger.info(f"[SYS-TEST] 💽 HF Cache Directory ({hf_cache}): {free_gb:.2f} GB Free")
        
        if free_gb < 80.0:
            logger.warning(f"[SYS-TEST] ⚠️ LOW DISK SPACE! You have {free_gb:.2f}GB free. The models (Gemma/Llama/Vicuna) require ~80GB. If not already downloaded, the pipeline will crash.")
        else:
            logger.info("[SYS-TEST] ✅ Storage space is sufficient.")
    except Exception as e:
        logger.warning(f"[SYS-TEST] ⚠️ Could not verify disk space: {e}")

    # ---------------------------------------------------------
    # 3. TEST NCCL DISTRIBUTED COMMUNICATION
    # ---------------------------------------------------------
    try:
        logger.info("[SYS-TEST] 🔌 Testing NCCL Distributed Communication (Port 29500)...")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        logger.info("[SYS-TEST] ✅ NCCL Backend initialized successfully.")
        dist.destroy_process_group()
    except Exception as e:
        logger.error(f"[SYS-TEST] ❌ NCCL Initialization failed: {e}")
        logger.info("[SYS-TEST] 💡 Hint: Port 29500 is blocked. Run 'sudo fuser -k 29500/tcp' to clear ghost processes.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 4. TEST VRAM FRAGMENTATION
    # ---------------------------------------------------------
    try:
        logger.info("[SYS-TEST] 🧱 Testing VRAM Allocation (Allocating 2GB per GPU)...")
        tensors = []
        for i in range(torch.cuda.device_count()):
            tensors.append(torch.empty(1024*1024*512, dtype=torch.float32, device=f"cuda:{i}")) # ~2GB
        logger.info("[SYS-TEST] ✅ VRAM Allocation successful.")
        del tensors
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"[SYS-TEST] ❌ VRAM Allocation failed (Fragmentation or Ghost Processes): {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 5. TEST TRUE P2P BANDWIDTH (1GB TENSOR TRANSFER)
    # ---------------------------------------------------------
    try:
        logger.info("[SYS-TEST] ⚡ Testing GPU P2P PCIe/NVLink Bandwidth (1GB Transfer)...")
        tensor_size_mb = 1024
        elements = (tensor_size_mb * 1024 * 1024) // 4  # float32 is 4 bytes
        
        # Warm up GPUs to wake them from idle state
        dummy = torch.randn(1024, device="cuda:0").to("cuda:1")
        torch.cuda.synchronize()
        
        # Generate 1GB tensor on GPU 0
        tensor_g0 = torch.randn(elements, dtype=torch.float32, device="cuda:0")
        
        # Setup CUDA timing events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        # Transfer the 1GB tensor to GPU 1
        tensor_g1 = tensor_g0.to("cuda:1")
        end_event.record()
        
        torch.cuda.synchronize()
        
        time_ms = start_event.elapsed_time(end_event)
        if time_ms > 0:
            bandwidth_gb_s = tensor_size_mb / time_ms # (MB / ms) converts exactly to (GB / s)
            logger.info(f"[SYS-TEST] ✅ P2P Transfer Complete! Bandwidth: {bandwidth_gb_s:.2f} GB/s ({time_ms:.2f} ms)")
            
            if bandwidth_gb_s < 2.0:
                logger.warning("[SYS-TEST] ⚠️ Warning: P2P bandwidth is extremely low (<2 GB/s). Tensor Parallelism might run slower than expected.")
        else:
            logger.info("[SYS-TEST] ✅ P2P Transfer Complete! (Time too small to measure bandwidth accurately)")

        del tensor_g0, tensor_g1, dummy
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"[SYS-TEST] ❌ P2P Bandwidth Test failed. GPUs cannot communicate properly: {e}")
        sys.exit(1)

    logger.info("[SYS-TEST] ✨ SYSTEM READY: All hyper-paranoid pre-flight checks passed flawlessly.")
    logger.info("="*60)
    sys.exit(0)

if __name__ == "__main__":
    test_distributed_env()