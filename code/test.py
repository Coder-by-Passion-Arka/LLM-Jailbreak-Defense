from vllm import LLM, SamplingParams

# 1. We use a tiny model (Qwen-1.5-0.5B) because Llama-2-7B is too big for 4GB.
# 2. We use 'quantization="awq"' to compress weights.
# 3. We enforce strict memory limits.

try:
    print("Attempting to load model into 4GB VRAM...")
    llm = LLM(
        model="Qwen/Qwen1.5-0.5B-Chat-AWQ",  # A tiny, compressed model
        quantization="awq",                  # 4-bit quantization
        dtype="float16",
        gpu_memory_utilization=0.95,         # Use 95% of your GPU
        enforce_eager=True                   # Save memory by disabling CUDA graphs
    )
    
    output = llm.generate("Hello, are you working?")
    print(output[0].outputs[0].text)

except Exception as e:
    print(f"\nCRITICAL FAILURE: {e}")
    print("Your GPU simply does not have enough memory to run vLLM.")