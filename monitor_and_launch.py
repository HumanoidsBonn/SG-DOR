#!/usr/bin/env python3
import subprocess
import time
import os

# Configuration
GPUS_TO_MONITOR = [0, 1, 2]
#KEYS_TO_RUN = ["S2S0", "S2H3", "S2H2", "S2H1", "S2L2", "S2L1", "S2L4", "S2L5", "S2S1", "S2L3"]
#KEYS_TO_RUN = ["S2H3", "S2L2", "S2L1", "S2L4", "S2L5", "S2S1", "S2L3"]
KEYS_TO_RUN = ["S2H1I16", "S2H2I16", "S2L3I16"]

    # = 10.0  # Minimum free memory to consider GPU "empty"
MAX_USED_MEM_GB = 1.0         # Maximum used memory to consider GPU "empty"
CHECK_INTERVAL_SEC = 30       # How often to poll nvidia-smi
WAIT_AFTER_LAUNCH_SEC = 120    # Give the model time to allocate memory before launching the next one

# Paths
WORKSPACE = "/home/user/menon/workspace/software/pepper_scene_graphs"

def get_used_gpu_memory():
    """Returns a dictionary mapping GPU ID to used memory in GB."""
    try:
        # Query used memory
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,nounits,noheader"
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        used_mem = {}
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            idx, mem = line.split(',')
            used_mem[int(idx.strip())] = float(mem.strip()) / 1024.0 # Convert MB to GB
        return used_mem
    except Exception as e:
        print(f"Error checking nvidia-smi: {e}")
        return {}

def launch_experiment(key: str, gpu_id: int):
    """Launches the experiment in a detached screen session."""
    session_name = key
    
    # We construct a bash command that:
    # 1. Source conda
    # 2. Activate environment
    # 3. Change directory
    # 4. Run python script
    # We use 'bash -c' so that conda activate works
    
    # Get the base path of conda to source it properly. 
    # Usually in ~/miniconda3/etc/profile.d/conda.sh or ~/.conda
    # Since we don't know the exact conda path, a safer way in bash scripts is `eval "$(conda shell.bash hook)"`
    
    cmd_sequence = f"""
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate pepper_train
cd {WORKSPACE}
python -u -m scripts.ablations.launch_oracle_ablation --gpu {gpu_id} --key {key}
# Keep the screen open if it crashes so we can read the traceback
echo 'Process finished or crashed. Press Enter to close.'
read
"""
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching {key} on GPU {gpu_id} inside screen session '{session_name}'")
    
    # We write the command to a temporary shell script because passing multiline 
    # complex strings directly into screen can be flaky.
    script_path = f"/tmp/launch_{key}.sh"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(cmd_sequence)
    os.chmod(script_path, 0o755)
    
    # Launch the script inside a detached screen session
    subprocess.run(["screen", "-dmS", session_name, script_path], check=True)

def main():
    print(f"Monitoring GPUs {GPUS_TO_MONITOR} for used memory < {MAX_USED_MEM_GB} GB")
    print(f"Queue: {KEYS_TO_RUN}")
    
    queue = list(KEYS_TO_RUN)
    
    while queue:
        # 1. Check GPU memory
        used_mem = get_used_gpu_memory()
        
        # 2. Find an available GPU
        available_gpu = None
        for gpu_id in GPUS_TO_MONITOR:
            if gpu_id in used_mem and used_mem[gpu_id] < MAX_USED_MEM_GB:
                available_gpu = gpu_id
                break
                
        # 3. Launch if available
        if available_gpu is not None:
            key_to_run = queue.pop(0)
            launch_experiment(key_to_run, available_gpu)
            
            if queue:
                print(f"Waiting {WAIT_AFTER_LAUNCH_SEC}s for memory to allocate before checking again...")
                time.sleep(WAIT_AFTER_LAUNCH_SEC)
        else:
            # 4. Wait and try again
            time.sleep(CHECK_INTERVAL_SEC)
            
    print("All tasks launched!")

if __name__ == "__main__":
    main()
