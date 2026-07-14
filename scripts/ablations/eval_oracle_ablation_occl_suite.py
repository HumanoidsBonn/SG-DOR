import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


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
    matches = sorted(runs_root.glob(f"*_{expected_suffix}"))
    if len(matches) == 1:
        return matches[0]
    return None


def _extract_scene_ids_from_sidecars(sidecar_dir: Path, n_first: int) -> List[int]:
    ids: Set[int] = set()
    for p in sorted(sidecar_dir.glob("scene_*_occl_cam_*.npz")):
        parts = p.name.split("_")
        if len(parts) < 2 or parts[0] != "scene":
            continue
        try:
            ids.add(int(parts[1]))
        except Exception:
            continue
    ids_sorted = sorted(ids)
    if not ids_sorted:
        raise SystemExit(f"[abort] no camera sidecars found in {sidecar_dir}")
    return ids_sorted[: min(int(n_first), len(ids_sorted))]


def _find_created_run_dir(runs_root: Path, exp_name: str) -> Optional[Path]:
    # train_oracle uses: run_name = <timestamp>_<exp_name>, and run_dir = runs_root/run_name
    matches = sorted(runs_root.glob(f"*_{exp_name}"), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="Ablation key to evaluate (A0, S2, ...)")
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
        "--out_dir",
        type=str,
        required=True,
        help="Directory to write the eval suite (single folder with per-variant sub-runs + a summary.json).",
    )
    ap.add_argument(
        "--first_n_scenes",
        type=int,
        required=True,
        help="Evaluate on the first N *available* camera sidecar scene ids (sorted by scene id).",
    )
    ap.add_argument(
        "--cam_sidecar_dir",
        type=str,
        default=None,
        help="Directory containing camera sidecars (scene_XXXXX_occl_cam_*.npz).",
    )
    ap.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="W&B mode for the eval runs",
    )
    ap.add_argument("--wandb_project", type=str, default="pepper_oracle", help="W&B project (override)")
    ap.add_argument("--wandb_group", type=str, default="oracle_eval_suite", help="W&B group for suite runs")
    ap.add_argument("--print_only", action="store_true", help="Print resolved commands and exit")
    args = ap.parse_args()

    suite_dir = Path(args.out_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    runs_root = suite_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    spec = _load_spec(Path(args.spec))
    run = _resolve_run(spec, str(args.key).strip())
    train_script = str(spec["train_script"])

    # Resolve trained checkpoint.
    train_runs_root = Path(args.train_runs_root)
    trained_run_dir = Path(args.run_dir) if args.run_dir else _auto_find_run_dir(train_runs_root, run["name"])
    if trained_run_dir is None:
        raise SystemExit(
            f"[abort] could not auto-detect run dir for key={run['key']} name={run['name']} under {train_runs_root}. "
            f"Pass --run_dir explicitly."
        )
    if not trained_run_dir.exists():
        raise SystemExit(f"[abort] trained run_dir does not exist: {trained_run_dir}")

    ckpt_name = "best_model.pt" if args.ckpt_kind == "best" else "last_model.pt"
    ckpt_path = trained_run_dir / ckpt_name
    if not ckpt_path.exists():
        raise SystemExit(f"[abort] missing checkpoint: {ckpt_path}")

    # Resolve camera sidecar dir (default heuristic).
    cam_sidecar_dir = None
    if args.cam_sidecar_dir:
        cam_sidecar_dir = Path(str(args.cam_sidecar_dir))
    else:
        guess = Path("datasets") / "ray_full800_ortho_persp_20260301_1747" / "occl_cam_sidecars"
        if guess.exists():
            cam_sidecar_dir = guess
    if cam_sidecar_dir is None or (not cam_sidecar_dir.exists()):
        raise SystemExit(
            "[abort] missing --cam_sidecar_dir and default guess not found. "
            "Pass --cam_sidecar_dir explicitly."
        )

    # Scene allowlist: first N available camera-sidecar scenes (sorted ids).
    scene_ids = _extract_scene_ids_from_sidecars(cam_sidecar_dir, int(args.first_n_scenes))
    if not scene_ids:
        raise SystemExit("[abort] no scene ids selected (unexpected).")
    scene_ids_path = suite_dir / f"scene_ids_first{int(args.first_n_scenes)}.json"
    scene_ids_txt = suite_dir / f"scene_ids_first{int(args.first_n_scenes)}.txt"
    scene_ids_path.write_text(json.dumps(scene_ids, indent=2) + "\n", encoding="utf-8")
    scene_ids_txt.write_text("\n".join(str(i) for i in scene_ids) + "\n", encoding="utf-8")

    suite_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _run_variant(tag: str, extra_args: List[str]) -> Tuple[str, Optional[Path], int]:
        exp_name = f"{suite_id}_{run['key']}_{tag}"
        cmd = [
            sys.executable,
            "-u",
            train_script,
            "--name",
            exp_name,
            "--runs_root",
            str(runs_root),
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
            "eval_suite",
            "--val_only",
            "--resume",
            str(ckpt_path),
        ] + list(run["args"]) + list(extra_args)

        print(_fmt_cmd(cmd), flush=True)
        if args.print_only:
            return exp_name, None, 0

        env = dict(os.environ)
        rc = subprocess.run(cmd, env=env).returncode
        created = _find_created_run_dir(runs_root, exp_name)
        return exp_name, created, int(rc)

    # Build all variants:
    # - camera ortho: jitter 0/1/2
    # - camera persp: jitter 0/1/2
    # - zbuf baseline: on the same scene ids
    variants: List[Tuple[str, List[str]]] = []
    for mode in ("cam_ortho", "cam_persp"):
        for jl in (0, 1, 2):
            tag = f"{mode}_j{jl}"
            variants.append(
                (
                    tag,
                    [
                        "--occl_gt_source",
                        mode,
                        "--occl_sidecar_dir",
                        str(cam_sidecar_dir),
                        "--occl_eval_scene_ids_file",
                        str(scene_ids_path),
                        "--occl_cam_jitter_level",
                        str(jl),
                    ],
                )
            )
    variants.append(
        (
            "zbuf",
            [
                "--occl_gt_source",
                "zbuf",
                "--occl_eval_scene_ids_file",
                str(scene_ids_path),
            ],
        )
    )

    manifest = {
        "suite_id": suite_id,
        "ablation_key": str(run["key"]),
        "trained_run_dir": str(trained_run_dir),
        "ckpt_kind": str(args.ckpt_kind),
        "ckpt_path": str(ckpt_path),
        "cam_sidecar_dir": str(cam_sidecar_dir),
        "first_n_scenes": int(args.first_n_scenes),
        "scene_ids_file": str(scene_ids_path),
        "scene_ids_count": int(len(scene_ids)),
        "scene_ids_min": int(scene_ids[0]),
        "scene_ids_max": int(scene_ids[-1]),
        "variants": [],
    }

    suite_ok = True
    for tag, extra in variants:
        exp_name, created_dir, rc = _run_variant(tag, extra)
        entry = {
            "tag": tag,
            "exp_name": exp_name,
            "returncode": int(rc),
            "run_dir": str(created_dir) if created_dir else None,
        }
        if rc != 0:
            suite_ok = False
        else:
            # Pull a few key metrics into the suite manifest if available.
            if created_dir is not None:
                mpath = created_dir / "val_metrics.json"
                if mpath.exists():
                    try:
                        mets = json.loads(mpath.read_text(encoding="utf-8"))
                        entry["val_metrics"] = {
                            k: mets.get(k)
                            for k in (
                                "val/occl_dir_best_f1",
                                "val/occl_dir_f1",
                                "val/occl_leaf_ndcg_at_5",
                                "val/occl_leaf_recall_at_3",
                                "val/occl_leaf_mass_at_3",
                                "val/occl_union_mae_high",
                                "val/occl_union_explain_w_pos_mean",
                            )
                        }
                    except Exception:
                        pass
        manifest["variants"].append(entry)

        # Update suite manifest after every run (so partial progress is visible if something crashes).
        (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        if rc != 0:
            raise SystemExit(f"[abort] variant failed: {tag} (rc={rc})")

    if suite_ok:
        print(f"[suite] complete. manifest: {suite_dir / 'suite_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

