import subprocess
import time
import os
import signal
from datetime import datetime

# Configuration
GPUS = [1, 2, 3]
MEMORY_THRESHOLD_MB = 4000  # Consider free if < 4GB used
POLL_INTERVAL_SEC = 60

EXPERIMENTS = [
    {
        "name": "oracle_v12_blind_xyz_normals",
        "cmd": (
            "python3 scripts/train_oracle.py --model_type res --node_feat_mode local "
            "--no_rgb --drop_shoots --no_geom_injection --reversed_curriculum "
            "--subgraph_enabled --subgraph_size 64 --batch_size 32 --lr 2e-3 "
            "--exp_name oracle_v12_blind_xyz_normals"
        )
    },
    {
        "name": "oracle_v12_blind_xyz_only",
        "cmd": (
            "python3 scripts/train_oracle.py --model_type res --node_feat_mode local "
            "--no_rgb --no_normals --drop_shoots --no_geom_injection --reversed_curriculum "
            "--subgraph_enabled --subgraph_size 64 --batch_size 32 --lr 2e-3 "
            "--exp_name oracle_v12_blind_xyz_only"
        )
    }
]

def get_gpu_memory(gpu_id):
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            encoding='utf-8'
        )
        return int(output.strip())
    except Exception as e:
        print(f"Error checking GPU {gpu_id}: {e}")
        return 999999

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def launch_experiment(gpu_id, exp):
    log_file = f"pepper_scene_graphs/training/logs/{exp['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    screen_cmd = (
        f"screen -dmS {exp['name']}_gpu{gpu_id} bash -c "
        f"\"export CUDA_VISIBLE_DEVICES={gpu_id}; export PYTHONUNBUFFERED=1; "
        f"conda run -n pepper_train {exp['cmd']} 2>&1 | tee {log_file}\""
    )
    log(f"Launching {exp['name']} on GPU {gpu_id}...")
    subprocess.run(screen_cmd, shell=True)

def main():
    pending = list(EXPERIMENTS)
    log(f"Starting Blind Oracle monitor. Target GPUs: {GPUS}. Pending: {len(pending)}")
    
    while pending:
        for gpu_id in GPUS:
            if not pending:
                break
                
            mem = get_gpu_memory(gpu_id)
            if mem < MEMORY_THRESHOLD_MB:
                exp = pending.pop(0)
                launch_experiment(gpu_id, exp)
                # Small sleep to ensure screen started
                time.sleep(5)
            else:
                # Still busy
                pass
        
        if pending:
            time.sleep(POLL_INTERVAL_SEC)

    log("All experiments launched. Exiting.")

if __name__ == "__main__":
    main()
