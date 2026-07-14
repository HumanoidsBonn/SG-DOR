import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _proc_exists(pid: int) -> bool:
    try:
        return Path(f"/proc/{int(pid)}").exists()
    except Exception:
        return False


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _collect(runs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not runs_root.exists():
        return rows
    for d in runs_root.iterdir():
        if not d.is_dir():
            continue
        prog = _read_json(d / "progress.json") or {}
        cfg = _read_json(d / "config.json") or {}
        pid = prog.get("pid") or cfg.get("pid")
        pid_i = int(pid) if isinstance(pid, (int, float, str)) and str(pid).isdigit() else None
        running = _proc_exists(pid_i) if pid_i is not None else False

        key = prog.get("ablation_key") or cfg.get("ablation_key")
        epoch = prog.get("epoch")
        phase = prog.get("phase")
        gpu = prog.get("cuda_visible_devices") or os.environ.get("CUDA_VISIBLE_DEVICES", "")

        best = (prog.get("best_monitor") or {})
        best_name = best.get("name")
        best_val = best.get("value")
        best_ep = best.get("epoch")

        # Prefer val metrics when available.
        val = prog.get("val") or {}
        train = prog.get("train") or {}
        occl_ndcg3 = val.get("val/occl_leaf_ndcg_at_3") or train.get("train/occl_leaf_ndcg_at_3")
        occl_dir_f1 = val.get("val/occl_dir_best_f1") or train.get("train/occl_dir_f1")
        edge_f1 = val.get("val/edge_exist_best_f1") or train.get("train/edge_exist_f1")

        rows.append(
            {
                "dir": d.name,
                "key": key,
                "gpu": gpu,
                "pid": pid_i,
                "running": running,
                "phase": phase,
                "epoch": epoch,
                "best_name": best_name,
                "best_val": best_val,
                "best_epoch": best_ep,
                "occl_ndcg3": occl_ndcg3,
                "occl_dir_f1": occl_dir_f1,
                "edge_f1": edge_f1,
                "mtime": d.stat().st_mtime,
            }
        )
    return rows


def _render(rows: List[Dict[str, Any]], limit: int) -> str:
    cols = [
        ("run_dir", "dir"),
        ("key", "key"),
        ("gpu", "gpu"),
        ("pid", "pid"),
        ("state", "running"),
        ("phase", "phase"),
        ("ep", "epoch"),
        ("best", "best_val"),
        ("best_ep", "best_epoch"),
        ("ndcg@3", "occl_ndcg3"),
        ("dir_f1", "occl_dir_f1"),
        ("edge_f1", "edge_f1"),
    ]

    # Format rows and compute widths.
    view: List[List[str]] = []
    for r in rows[:limit]:
        view.append(
            [
                str(r.get("dir", "")),
                _fmt(r.get("key")),
                _fmt(r.get("gpu")),
                _fmt(r.get("pid")),
                "RUN" if r.get("running") else "DONE",
                _fmt(r.get("phase")),
                _fmt(r.get("epoch")),
                _fmt(r.get("best_val")),
                _fmt(r.get("best_epoch")),
                _fmt(r.get("occl_ndcg3")),
                _fmt(r.get("occl_dir_f1")),
                _fmt(r.get("edge_f1")),
            ]
        )

    widths = [len(h) for h, _ in cols]
    for row in view:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(items: List[str]) -> str:
        return "  ".join(items[i].ljust(widths[i]) for i in range(len(items)))

    header = fmt_row([h for h, _ in cols])
    sep = fmt_row(["-" * w for w in widths])
    body = "\n".join(fmt_row(r) for r in view)
    return "\n".join([header, sep, body]) + ("\n" if body else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="training/oracle", help="Root directory containing run folders")
    ap.add_argument("--limit", type=int, default=20, help="Max number of runs to show")
    ap.add_argument("--watch", type=float, default=0.0, help="Refresh interval seconds (0 disables watch mode)")
    ap.add_argument(
        "--sort",
        type=str,
        default="mtime",
        choices=["mtime", "epoch", "key"],
        help="Sort key (descending for mtime/epoch, ascending for key)",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root)

    while True:
        rows = _collect(runs_root)
        if args.sort == "mtime":
            rows.sort(key=lambda r: float(r.get("mtime", 0.0)), reverse=True)
        elif args.sort == "epoch":
            rows.sort(key=lambda r: float(r.get("epoch") or 0.0), reverse=True)
        else:
            rows.sort(key=lambda r: str(r.get("key") or ""))

        os.system("clear")
        print(_render(rows, int(args.limit)), flush=True)
        if float(args.watch) <= 0.0:
            break
        time.sleep(float(args.watch))


if __name__ == "__main__":
    main()
