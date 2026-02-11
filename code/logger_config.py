import logging
import sys
import os

# --- CONFIGURATION DIRECTORIES ---
LOGS_DIR = "./logs"
RESULT_DIR = "./results"
OUTPUT_DIR = "./model_responses"

# Ensure directories exist
for directory in [LOGS_DIR, RESULT_DIR, OUTPUT_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

TERMINAL_OUTPUT_FILE = os.path.join(LOGS_DIR, "terminal_output.txt")

# --- CUSTOM DUAL LOGGER ---
class DualLogger(object):
    """Mirrors stdout/stderr to a file + terminal."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logger(name="JailbreakPipeline"):
    # 1. Redirect stdout/stderr globally
    sys.stdout = DualLogger(TERMINAL_OUTPUT_FILE)
    sys.stderr = sys.stdout 

    # 2. Configure Python's logging module
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # File Handler (Detailed execution log)
        file_handler = logging.FileHandler(os.path.join(LOGS_DIR, "execution.log"), mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Initialize shared logger
logger = setup_logger()