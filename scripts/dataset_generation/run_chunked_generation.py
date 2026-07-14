#!/usr/bin/env python3

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args():
    p = argparse.ArgumentParser(description="Chunked driver for scene_generation_obb.py")
    p.add_argument("--scene", required=True, help="Path to input .blend file")
    p.add_argument("--scenes-root", required=True, help="Root directory to store runs")
    p.add_argument("--run-name", default="industrial_v1", help="Base name for this run")
    p.add_argument("--chunk-size", type=int, default=50, help="Scenes per chunk")
    p.add_argument("--total-scenes", type=int, required=True, help="Total scenes to generate")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel blenderproc workers")
    p.add_argument("--bproc-bin", default="blenderproc", help="blenderproc executable")
    p.add_argument("--script", default="scene_generation_obb.py", help="Path to scene generation script")
    p.add_argument("--config-json", default=None, help="Path to a JSON file containing CONFIG overrides")
    p.add_argument("--extract-pc", action="store_true", help="Enable integrated point cloud extraction")
    p.add_argument("--no-blend", action="store_true", help="Skip .blend saving")
    p.add_argument("--voxel-size", type=float, default=0.002, help="PC voxel size")
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Extra args passed to scene_generation_obb.py")
    return p.parse_args()


def run_chunk(bproc_bin, script, scene, start, count, data_dir, extra_args):
    """Executes a single chunk generation."""
    t0 = time.time()
    # Log files are kept in a separate folder to avoid polluting the data folder
    log_dir = data_dir.parent / "logs"
    chunk_log = log_dir / f"chunk_{start:05d}.log"
    
    cmd = [
        str(bproc_bin),
        "run",
        str(script),
        str(scene),
        "--scene_start",
        str(start),
        "--scene_count",
        str(count),
        "--data_dir",
        str(data_dir),
        "--graphs_out",
        str(data_dir.parent / "chunks" / f"chunk_{start:05d}.json"),
    ]
    if extra_args:
        cmd.extend([str(a) for a in extra_args])
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Launching Chunk {start}-{start+count-1} (see logs/chunk_{start:05d}.log)")
    
    # Stream subprocess output to both console and log file (tee-style).
    with open(chunk_log, "w") as f_log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f_log.write(line)
            f_log.flush()
        res = proc.wait()
    
    elapsed = time.time() - t0
    if res != 0:
        print(f"FAILED: Chunk {start}-{start+count-1} after {elapsed:.1f}s. Check {chunk_log}")
        return False
    else:
        print(f"SUCCESS: Chunk {start}-{start+count-1} in {elapsed:.1f}s")
        return True


def main():
    args = parse_args()
    driver_dir = Path(__file__).parent
    script_path = (driver_dir / args.script).resolve()
    if not script_path.exists():
        script_path = Path(args.script).resolve()
        
    t0_run = time.time()
    if not script_path.exists():
        print(f"Error: Script {args.script} not found.")
        sys.exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = Path(args.scenes_root) / f"{args.run_name}_{timestamp}"
    data_dir = run_dir / "data"
    chunk_dir = run_dir / "chunks"
    log_dir = run_dir / "logs"
    
    for d in [data_dir, chunk_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    final_json = run_dir / "scene_graphs.json"
    params_path = run_dir / f"params_{args.run_name}.json"

    if args.config_json:
        src_cfg = Path(args.config_json).resolve()
        params_path.write_bytes(src_cfg.read_bytes())
        print(f"Copied config override to {params_path}")

    num_chunks = math.ceil(args.total_scenes / args.chunk_size)
    common_extra = list(args.extra_args) if args.extra_args else []
    if args.config_json:
        common_extra.extend(["--config_override", str(params_path)])
    if args.extract_pc:
        common_extra.extend(["--extract-pc", "--voxel-size", str(args.voxel_size)])
    if args.no_blend:
        common_extra.append("--no-blend")

    print(f"Starting Industrial Generation: {args.total_scenes} scenes, {args.workers} workers.")
    print(f"Output: {run_dir}")
    print("Files will appear in data/ as each scene is completed by a worker.")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i in range(num_chunks):
            start = i * args.chunk_size
            count = min(args.chunk_size, args.total_scenes - start)
            
            futures.append(executor.submit(
                run_chunk, args.bproc_bin, script_path, args.scene, 
                start, count, data_dir, common_extra
            ))
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker-level exception: {e}")

    # Final Aggregation (optional, but nice for provenance)
    merged_data = []
    chunk_jsons = list(chunk_dir.glob("*.json"))
    print(f"Post-processing: Aggregating {len(chunk_jsons)} chunk files...")
    for cj in chunk_jsons:
        try:
            with open(cj, "r", encoding="utf-8") as f:
                merged_data.extend(json.load(f))
        except Exception as e:
            print(f"Warning: Failed to load {cj}: {e}")

    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)
    
    t_run_total = time.time() - t0_run
    print(f"--- COMPLETE ---")
    print(f"Total execution time: {t_run_total:.2f}s")


if __name__ == "__main__":
    main()
