import time
import os
import re
import sys

# Paths to monitor
LOGS = {
    "Decoupled v6 (Merged)": "multi_head_decoupled_v6.log",
    "Geometry v4 (Merged)": "multi_head_geometry_v4.log"
}

TARGET_EPOCH = 50

def parse_last_epoch(content):
    """Find the highest epoch number in the log."""
    matches = re.findall(r"Epoch (\d+)/50", content)
    if matches:
        return int(matches[-1])
    return 0

def is_done(content):
    """Check if the final epoch has finished validating."""
    return f"Epoch {TARGET_EPOCH}/50" in content and "Confusion Matrix" in content.split(f"Epoch {TARGET_EPOCH}/50")[-1]

def monitor():
    print(f"🚀 Antigravity Watchdog: Monitoring for Epoch {TARGET_EPOCH} completion...")
    finished = set()
    
    while True:
        for name, log_file in LOGS.items():
            if name in finished:
                continue
                
            if not os.path.exists(log_file):
                continue
                
            with open(log_file, 'r') as f:
                content = f.read()
            
            epoch = parse_last_epoch(content)
            
            if is_done(content):
                print(f"✅ {name}: COMPLETE!")
                finished.add(name)
            else:
                # Use \r to update the same line
                sys.stdout.write(f"\r⏳ {name}: Epoch {epoch}/{TARGET_EPOCH} ...")
                sys.stdout.flush()
        
        if len(finished) == len(LOGS):
            print("\n\n🎉 ALL EXPERIMENTS COMPLETE!")
            # Create a flag file for the IDE agent to pick up
            with open("TRAINING_SUCCESS.flag", "w") as f:
                f.write("Training finished at " + time.ctime())
            break
            
        time.sleep(30) # Poll every 30 seconds

if __name__ == "__main__":
    monitor()
