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


def _fmt_cmd(cmd: List[str]) -> str:
    import shlex

    return " ".join(shlex.quote(c) for c in cmd)


def _auto_find_run_dir(runs_root: Path, expected_suffix: str) -> Optional[Path]:
    # run dirs are timestamp-prefixed: 20260218_193716_<name>
    matches = sorted(runs_root.glob(f"*_{expected_suffix}"))
    if len(matches) == 1:
        return matches[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="Ablation key to evaluate (A0, A1, ...)")
    ap.add_argument(
        "--spec",
        type=str,
        default="scripts/ablations/oracle_ablations.json",
        help="Path to ablation spec JSON",
    )
    ap.add_argument(
        "--train_runs_root",
        type=str,
        default="training/oracle",
        help="Root where the *trained* runs live (used to find checkpoints/configs)",
    )
    ap.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Explicit trained run dir to evaluate (overrides auto-detect).",
    )
    ap.add_argument(
        "--ckpt_kind",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Which checkpoint file to evaluate from the trained run dir",
    )
    ap.add_argument(
        "--eval_runs_root",
        type=str,
        default="/tmp/oracle_eval",
        help="Root for the new eval-only run folder that train_oracle.py will create",
    )
    ap.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="W&B mode for the eval-only run",
    )
    ap.add_argument("--wandb_project", type=str, default="pepper_oracle", help="W&B project (override)")
    ap.add_argument("--wandb_group", type=str, default="oracle_eval", help="W&B group for eval-only runs")
    ap.add_argument("--print_only", action="store_true", help="Print resolved command and exit")
    ap.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Extra args passed through to train_oracle.py (must come last; everything after is forwarded).",
    )
    args = ap.parse_args()

    spec = _load_spec(Path(args.spec))
    run = _resolve_run(spec, str(args.key).strip())
    train_script = str(spec["train_script"])

    train_runs_root = Path(args.train_runs_root)
    run_dir = Path(args.run_dir) if args.run_dir else _auto_find_run_dir(train_runs_root, run["name"])
    if run_dir is None:
        raise SystemExit(
            f"[abort] could not auto-detect run dir for key={run['key']} name={run['name']} under {train_runs_root}. "
            f"Pass --run_dir explicitly."
        )
    if not run_dir.exists():
        raise SystemExit(f"[abort] run_dir does not exist: {run_dir}")

    ckpt_name = "best_model.pt" if args.ckpt_kind == "best" else "last_model.pt"
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        raise SystemExit(f"[abort] missing checkpoint: {ckpt_path}")

    eval_name = f"{run['name']}_EVAL_{args.ckpt_kind}"
    cmd = [
        sys.executable,
        "-u",
        train_script,
        "--name",
        eval_name,
        "--runs_root",
        str(args.eval_runs_root),
        "--wandb_mode",
        str(args.wandb_mode),
        "--wandb_project",
        str(args.wandb_project),
        "--wandb_group",
        str(args.wandb_group),
        "--ablation_key",
        str(run["key"]),
        "--wandb_tag",
        str(run["key"]),
        "--wandb_tag",
        "eval_only",
        "--val_only",
        "--resume",
        str(ckpt_path),
    ] + list(run["args"])

    if args.extra_args:
        cmd += list(args.extra_args)

    print(_fmt_cmd(cmd), flush=True)
    if args.print_only:
        return

    env = dict(os.environ)
    rc = subprocess.run(cmd, env=env).returncode
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
