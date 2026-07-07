import subprocess
import sys
import time

def run_pipeline_sequence(commands):
    """
    Executes a sequence of arguments against run_pipeline.sh.
    """
    base_cmd = ["bash", "run_pipeline.sh"]
    
    print(f"🚀 Initializing Pipeline Orchestration at {time.strftime('%H:%M:%S')}")
    print(f"📋 Sequence length: {len(commands)} steps\n")

    for i, cmd_args in enumerate(commands, 1):
        # Flatten the command: base + user args
        full_command = base_cmd + cmd_args.split()
        
        print(f"--- STEP {i}/{len(commands)} ---")
        print(f"⚙️  Executing: {' '.join(full_command)}")
        
        try:
            # We use Popen to stream the output to the terminal in real-time
            # This is critical so you don't lose visibility during long HTS training
            process = subprocess.Popen(
                full_command,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )
            
            # Wait for this specific step to finish before starting the next
            return_code = process.wait()
            
            if return_code == 0:
                print(f"✅ Step {i} completed successfully.\n")
            else:
                print(f"❌ Step {i} failed with exit code {return_code}.")
                print("🛑 Aborting remaining sequence to prevent corrupted data.")
                sys.exit(return_code)
                
        except KeyboardInterrupt:
            print("\n👋 Orchestration interrupted by user. Killing subprocess...")
            process.terminate()
            sys.exit(1)
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")
            sys.exit(1)

    print(f"🏁 All steps completed at {time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    # --- YOUR COMMANDS HERE ---
    # Put exactly what you would type after 'bash run_pipeline.sh'
    to_be_executed = [
        "--train-hts all",    # Step 1: Train the calibrated manifolds
        "--infer all",        # Step 2: Run Red-Teaming (ASR)
        "--eval-fpr all"      # Step 3: Run Utility Testing (FPR)
    ]

    run_pipeline_sequence(to_be_executed)