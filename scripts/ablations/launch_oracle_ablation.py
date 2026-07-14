import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _load_spec(path: Path) -> Dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "train_script" not in obj or "base_args" not in obj or "runs" not in obj:
        raise ValueError(f"invalid spec file (missing train_script/base_args/runs): {path}")
    return obj


def _remove_flag(tokens: List[str], flag: str) -> List[str]:
    return [t for t in tokens if t != flag]


def _add_flag(tokens: List[str], flag: str) -> List[str]:
    if flag in tokens:
        return tokens
    return tokens + [flag]


def _set_arg(tokens: List[str], flag: str, value: str) -> List[str]:
    out = list(tokens)
    for i, t in enumerate(out):
        if t == flag:
            # Replace following value if present, else append.
            if i + 1 < len(out) and not out[i + 1].startswith("--"):
                out[i + 1] = str(value)
                return out
            out.insert(i + 1, str(value))
            return out
    out += [flag, str(value)]
    return out


def _resolve_run(spec: Dict, key: str) -> Dict:
    runs = spec["runs"]
    if key not in runs:
        raise KeyError(f"unknown key '{key}'. available: {', '.join(sorted(runs.keys()))}")
    base_tokens = list(spec["base_args"])
    run = runs[key]

    # Apply token-level edits (no argparse knowledge required).
    for flg in run.get("remove_flags", []) or []:
        base_tokens = _remove_flag(base_tokens, flg)
    for flg in run.get("add_flags", []) or []:
        base_tokens = _add_flag(base_tokens, flg)
    for flg, val in (run.get("set_args", {}) or {}).items():
        base_tokens = _set_arg(base_tokens, flg, str(val))

    name = str(run["name"])
    family = str(run.get("family", ""))
    return {"key": key, "name": name, "family": family, "args": base_tokens}


def _gpu_free_mem_gb(gpu_id: int) -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        ).strip()
        if not out:
            return None
        mb = float(out.splitlines()[0].strip())
        return mb / 1024.0
    except Exception:
        return None


def _fmt_cmd(cmd: List[str]) -> str:
    # Minimal safe quoting for terminal copy/paste.
    import shlex

    return " ".join(shlex.quote(c) for c in cmd)


def _run_subprocess(cmd: List[str], env: Dict[str, str]) -> int:
    p = subprocess.run(cmd, env=env)
    return int(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", action="append", required=True, help="Ablation key (repeatable), e.g. --key A0 --key L1")
    ap.add_argument(
        "--spec",
        type=str,
        default="scripts/ablations/oracle_ablations.json",
        help="Path to ablation spec JSON",
    )
    ap.add_argument("--gpu", type=int, default=None, help="GPU id (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--wandb_group", type=str, default="oracle_ablations_v1", help="W&B group for all launched runs")
    ap.add_argument("--wandb_project", type=str, default="pepper_oracle", help="W&B project (override)")
    ap.add_argument(
        "--extra_tag",
        action="append",
        default=None,
        help="Extra W&B tag(s) to add (repeatable).",
    )
    ap.add_argument("--no_preflight", action="store_true", help="Skip smoke preflight")
    ap.add_argument("--smoke_steps", type=int, default=2, help="Train batches for smoke test (per epoch)")
    ap.add_argument("--min_free_gb", type=float, default=2.0, help="Abort if GPU free memory is below this (if --gpu set)")
    ap.add_argument("--seed", type=int, default=None, help="Override --seed for the run(s) (default: spec value)")
    ap.add_argument("--print_only", action="store_true", help="Print resolved command(s) and exit")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    spec = _load_spec(spec_path)
    train_script = str(spec["train_script"])

    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        free_gb = _gpu_free_mem_gb(int(args.gpu))
        if free_gb is not None:
            print(f"[gpu] id={args.gpu} free_mem_gb={free_gb:.2f}", flush=True)
            if float(free_gb) < float(args.min_free_gb):
                raise SystemExit(
                    f"[abort] GPU {args.gpu} has only {free_gb:.2f} GB free (< {args.min_free_gb:.2f} GB)."
                )
        else:
            print("[gpu] nvidia-smi query failed; continuing without free-mem check.", flush=True)

    keys = []
    for k in args.key:
        if not k:
            continue
        keys.append(str(k).strip())

    for key in keys:
        run = _resolve_run(spec, key)
        if key not in run["name"]:
            raise SystemExit(f"[abort] spec name for key={key} does not include key: {run['name']}")

        run_args = list(run["args"])
        if args.seed is not None:
            run_args = _set_arg(run_args, "--seed", str(args.seed))

        # Inject W&B metadata so the web UI is readable/filterable.
        injected = [
            "--wandb_project",
            str(args.wandb_project),
            "--wandb_group",
            str(args.wandb_group),
            "--ablation_key",
            str(key),
            "--wandb_tag",
            str(key),
        ]
        if run.get("family"):
            injected += ["--wandb_tag", str(run["family"])]
        for t in args.extra_tag or []:
            injected += ["--wandb_tag", str(t)]

        cmd = [sys.executable, "-u", train_script, "--name", run["name"]] + injected + run_args

        print(f"\n=== {key} ===", flush=True)
        print(_fmt_cmd(cmd), flush=True)
        if args.print_only:
            continue

        if not args.no_preflight:
            smoke_root = Path("~/tmp/oracle_smoke")
            smoke_root.mkdir(parents=True, exist_ok=True)
            smoke_cmd = list(cmd)
            smoke_cmd = _set_arg(smoke_cmd, "--wandb_mode", "disabled")
            smoke_cmd = _set_arg(smoke_cmd, "--runs_root", str(smoke_root))
            smoke_cmd = _set_arg(smoke_cmd, "--epochs", "1")
            smoke_cmd = _set_arg(smoke_cmd, "--max_train_batches", str(max(1, int(args.smoke_steps))))
            smoke_cmd = _set_arg(smoke_cmd, "--stage_occl_epoch", "0")
            smoke_cmd = _set_arg(smoke_cmd, "--stage_gnn_epoch", "0")
            smoke_cmd = _set_arg(smoke_cmd, "--stage_reg_epoch", "0")
            smoke_cmd = _set_arg(smoke_cmd, "--val_interval", "999999")
            smoke_cmd = _set_arg(smoke_cmd, "--log_interval", "1")
            smoke_cmd = _set_arg(smoke_cmd, "--name", f"{run['name']}_SMOKE")

            print(f"[preflight] smoke test ({args.smoke_steps} batches)", flush=True)
            rc = _run_subprocess(smoke_cmd, env=env)
            if rc != 0:
                raise SystemExit(f"[abort] smoke test failed for key={key} (rc={rc})")
            print("[preflight] OK", flush=True)

        rc = _run_subprocess(cmd, env=env)
        if rc != 0:
            raise SystemExit(f"[abort] run failed for key={key} (rc={rc})")


if __name__ == "__main__":
    main()

