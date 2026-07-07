# ./logger_config.py

import logging
import sys
import os
import traceback
import faulthandler
import threading
import warnings
import time 

# =====================================================================
# DYNAMIC SCRIPT & MODEL ROUTING
# Intercepts the script name and target model to create perfectly isolated logs
# =====================================================================
# Identify which script is running (e.g., 'dataset_builder', 'train_hts', 'pipeline')
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
if not script_name or script_name == "-c":
    script_name = "interactive_session"

current_model = "global_execution"
if "--model" in sys.argv:
    try:
        idx = sys.argv.index("--model")
        current_model = sys.argv[idx + 1].replace("/", "_").replace("\\", "_")
    except IndexError:
        pass

# --- CONFIGURATION DIRECTORIES ---
LOGS_DIR = "./logs"
RESULT_DIR = "./results"
RESPONSE_DIR = "./model_responses"

# Ensure directories exist
for directory in [LOGS_DIR, RESULT_DIR, RESPONSE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- ISOLATED MODEL LOG FILES ---
TERMINAL_OUTPUT_FILE = os.path.join(LOGS_DIR, f"terminal_{script_name}_{current_model}.txt")
EXECUTION_LOG_FILE = os.path.join(LOGS_DIR, f"execution_{script_name}_{current_model}.log")

# =====================================================================
# FILE CLEARING MECHANISM (Wipes old logs for the current execution)
# =====================================================================
for file in [TERMINAL_OUTPUT_FILE, EXECUTION_LOG_FILE]: 
    with open(file, "w", encoding='utf-8') as f:
        f.write(f'{"="*70}\n')
        f.write(f'🚀 Script Module: {script_name}.py\n')
        f.write(f'🎯 Target Context: {current_model}\n')
        f.write(f'🕒 Execution Loop Started At: {time.ctime()}\n')
        f.write(f'{"="*70}\n\n')
                
# --- CUSTOM DUAL LOGGER (INFINITE-LOOP PROOF) ---
class DualLogger(object):
    """Mirrors stdout/stderr to a file + terminal without recursion."""
    def __init__(self, filename):
        # CRITICAL FIX 1: Bind explicitly to the immutable OS terminal stream.
        # This prevents the object from recursively calling itself.
        self.terminal = sys.__stdout__
        
        # Using 'a' (append) to prevent overwriting if multiple processes spawn
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        # Note: We do NOT force self.flush() here. tqdm calls write() 
        # thousands of times per second. Forcing disk sync here would throttle GPUs.

    def flush(self):
        # Let Python handle the memory buffers
        self.terminal.flush()
        self.log.flush()
        
        # CRITICAL FIX 2: We silently sync to disk. 
        # We MUST NOT use print() here, or it loops back to sys.stdout.
        try:
            os.fsync(self.log.fileno()) 
        except OSError:
            pass

def setup_logger(name="JailbreakPipeline"):
    # 1. Redirect stdout/stderr globally
    sys.stdout = DualLogger(TERMINAL_OUTPUT_FILE)
    sys.stderr = sys.stdout 

    # 2. Configure Python's logging module
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # Set to DEBUG to capture deep diagnostic info
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # ADD STREAM HANDLER FOR TERMINAL OUTPUT: explicitly tells the logger to print to the console 
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # File Handler (Detailed execution log)
        file_handler = logging.FileHandler(EXECUTION_LOG_FILE, mode='a', encoding='utf-8')
        
        # Enhanced Formatter: Now includes Process Name, Thread Name, and exact Line Numbers
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | [%(processName)s:%(threadName)s] | %(module)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 3. Capture Background Warnings (e.g., Deprecation, Resource Warnings)
    logging.captureWarnings(True)

    # =====================================================================
    # ADVANCED ERROR TRAPPING
    # =====================================================================

    # 4. Global Uncaught Exception Hook
    # Traps any error that slips past your try/except blocks
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Ignore Ctrl+C interrupts so they don't clog the logs
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "[LOGGER] 🔥 CRITICAL UNHANDLED EXCEPTION 🔥", 
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = global_exception_handler

    # 5. Threading Exception Hook
    # Traps errors occurring in background workers (like vLLM token streaming threads)
    def thread_exception_handler(args):
        logger.critical(
            f"[LOGGER] 🔥 CRITICAL THREAD CRASH [{args.thread.name}] 🔥", 
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )

    threading.excepthook = thread_exception_handler

    # 6. C-Level Fault Handler (The Ultimate Failsafe)
    # vLLM and PyTorch run on C++/CUDA. If they crash (Segfault, Bus Error, Abort), 
    # Python dies instantly without triggering exceptions.
    # faulthandler intercepts OS-level kill signals and dumps the trace to our file.
    try:
        fault_log_file = open(TERMINAL_OUTPUT_FILE, "a", encoding='utf-8')
        faulthandler.enable(file=fault_log_file, all_threads=True)
        logger.info("[LOGGER] ✅ OS faulthandler enabled to capture C-level crashes.")
    except Exception as e:
        logger.error(f"[LOGGER] ❌ Failed to enable faulthandler: {e}")
        logger.debug(traceback.format_exc())

    return logger

# Initialize shared logger
logger = setup_logger()