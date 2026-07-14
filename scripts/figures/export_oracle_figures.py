#!/usr/bin/env python3
"""
Export thesis-ready Oracle experiment figures from local run directories.

Key design points:
- Works *offline* by reading W&B's binary transaction log (run-*.wandb).
- Produces uniform, paper-friendly matplotlib figures (PDF/PNG) with many knobs.

Typical usage:
  /home/user/muellergoldingenn0/miniconda3/envs/bproc/bin/python \\
    -m scripts.figures.export_oracle_figures \\
    --out_dir latex_project/figures/oracle \\
    --export training_curves \\
    --training_curve_keys A0

Notes:
- If you're missing wandb/plot deps in your environment, use the `bproc` env.
- The script discovers run directories from scripts/ablations/oracle_ablations.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _as_list(x: str | list[str] | None) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [s for s in re.split(r"[,\s]+", x.strip()) if s]


def _safe_json_loads(s: str) -> Any:
    # wandb stores scalar values as JSON (numbers, strings, null, etc.).
    # Some edge cases may contain non-JSON (e.g., "nan") so we guard.
    try:
        return json.loads(s)
    except Exception:
        # Try a tiny subset of "relaxed" parsing for NaN/Inf.
        sl = s.strip().lower()
        if sl in {"nan", "+nan", "-nan"}:
            return float("nan")
        if sl in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if sl in {"-inf", "-infinity"}:
            return float("-inf")
        raise


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass
class StyleConfig:
    # --- Typography ---
    font_family: str = "DejaVu Serif"
    font_size: float = 10.0
    axes_title_size: float = 11.0
    axes_label_size: float = 10.0
    tick_label_size: float = 9.0
    legend_font_size: float = 9.0

    # --- Lines / markers ---
    line_width: float = 1.6
    marker_size: float = 4.0
    marker_every: int = 1  # 1=mark all points, >1=mark every Nth point

    # --- Layout ---
    figure_dpi: int = 150
    save_dpi: int = 300
    constrained_layout: bool = True

    # --- Axes styling ---
    grid: bool = True
    grid_alpha: float = 0.25
    grid_linewidth: float = 0.6
    spine_linewidth: float = 0.8
    legend_frameon: bool = False

    # --- Colors ---
    # Named colors used across plots; user can override via --style_json.
    colors: dict[str, str] = field(
        default_factory=lambda: {
            "A0": "#1f77b4",  # blue
            "A1": "#ff7f0e",  # orange
            "baseline": "#1f77b4",
            "compare": "#ff7f0e",
            "low": "#2ca02c",  # green
            "mid": "#d62728",  # red
            "high": "#9467bd",  # purple
            "missing": "#aaaaaa",
        }
    )

    # --- Smoothing / transforms ---
    smooth_window: int = 1  # 1 disables smoothing; >1 uses rolling mean

    # --- Export formats ---
    formats: list[str] = field(default_factory=lambda: ["pdf"])

    def update_from_dict(self, d: dict[str, Any]) -> None:
        for k, v in d.items():
            if k == "colors" and isinstance(v, dict):
                self.colors.update({str(kk): str(vv) for kk, vv in v.items()})
                continue
            if not hasattr(self, k):
                continue
            setattr(self, k, v)


def apply_mpl_style(style: StyleConfig) -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": style.figure_dpi,
            "savefig.dpi": style.save_dpi,
            "font.family": style.font_family,
            "font.size": style.font_size,
            "axes.titlesize": style.axes_title_size,
            "axes.labelsize": style.axes_label_size,
            "xtick.labelsize": style.tick_label_size,
            "ytick.labelsize": style.tick_label_size,
            "legend.fontsize": style.legend_font_size,
            "axes.grid": style.grid,
            "grid.alpha": style.grid_alpha,
            "grid.linewidth": style.grid_linewidth,
            "axes.linewidth": style.spine_linewidth,
            "lines.linewidth": style.line_width,
            "lines.markersize": style.marker_size,
            # Embed fonts (good for thesis/PDF pipelines)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _rolling_mean(xs: list[float], w: int) -> list[float]:
    if w <= 1:
        return xs
    out: list[float] = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        window = xs[lo : i + 1]
        out.append(sum(window) / max(1, len(window)))
    return out


@dataclass(frozen=True)
class RunInfo:
    key: str
    name: str
    family: str
    run_dir: Path
    wandb_dir: Path  # .../wandb/wandb/run-...
    wandb_file: Path  # run-<id>.wandb


def load_ablations_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def discover_run_dir(training_root: Path, run_name: str) -> Path | None:
    # Run directories are named like:
    #   20260218_193716_oracle_ablate_A0_baseline_no_edl
    pattern = f"*_{run_name}"
    matches = [p for p in training_root.glob(pattern) if p.is_dir()]
    if not matches:
        return None
    matches_sorted = sorted(matches, key=lambda p: p.name)
    if len(matches_sorted) > 1:
        _eprint(
            f"[warn] multiple run dirs match {run_name!r}; using latest: {matches_sorted[-1].as_posix()}"
        )
    return matches_sorted[-1]


def discover_wandb_file(run_dir: Path) -> tuple[Path, Path] | None:
    # Prefer wandb/wandb/latest-run, which is a symlink to run-* folder.
    base = run_dir / "wandb" / "wandb"
    latest = base / "latest-run"
    if latest.exists():
        wandb_files = sorted(latest.glob("run-*.wandb"))
        if len(wandb_files) == 1:
            return latest, wandb_files[0]
        if len(wandb_files) > 1:
            _eprint(
                f"[warn] multiple .wandb files in {latest.as_posix()}, using {wandb_files[-1].name}"
            )
            return latest, wandb_files[-1]

    # Fallback: search run-* directories.
    for run_subdir in sorted(base.glob("run-*")):
        wandb_files = sorted(run_subdir.glob("run-*.wandb"))
        if wandb_files:
            if len(wandb_files) > 1:
                _eprint(
                    f"[warn] multiple .wandb files in {run_subdir.as_posix()}, using {wandb_files[-1].name}"
                )
            return run_subdir, wandb_files[-1]

    return None


def build_run_infos(
    ablations_json: Path,
    training_root: Path,
    keys: list[str] | None,
    allow_missing: bool,
) -> dict[str, RunInfo]:
    ab = load_ablations_json(ablations_json)
    runs: dict[str, Any] = ab.get("runs", {})

    if keys:
        wanted = set(keys)
        runs = {k: v for k, v in runs.items() if k in wanted}

    out: dict[str, RunInfo] = {}
    for key, spec in runs.items():
        name = spec.get("name", "")
        family = spec.get("family", "unknown")
        if not name:
            if allow_missing:
                _eprint(f"[warn] missing name for key={key}, skipping")
                continue
            raise ValueError(f"missing run name for key {key}")

        run_dir = discover_run_dir(training_root, name)
        if run_dir is None:
            msg = f"missing run_dir for key={key} name={name!r} under {training_root.as_posix()}"
            if allow_missing:
                _eprint(f"[warn] {msg}")
                continue
            raise FileNotFoundError(msg)

        wandb = discover_wandb_file(run_dir)
        if wandb is None:
            msg = f"missing wandb .wandb file under {run_dir.as_posix()}"
            if allow_missing:
                _eprint(f"[warn] {msg}")
                continue
            raise FileNotFoundError(msg)

        wandb_dir, wandb_file = wandb
        out[key] = RunInfo(
            key=key,
            name=name,
            family=family,
            run_dir=run_dir,
            wandb_dir=wandb_dir,
            wandb_file=wandb_file,
        )

    return out


def load_progress(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "progress.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_history_df(
    run: RunInfo,
    *,
    cache_dir: Path | None,
    allow_incomplete: bool,
) -> "pd.DataFrame":
    import pandas as pd

    wandb_file = run.wandb_file
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        st = wandb_file.stat()
        cache_path = cache_dir / f"{run.key}__{wandb_file.stem}__{st.st_size}_{int(st.st_mtime)}.pkl"
        if cache_path.exists():
            try:
                return pd.read_pickle(cache_path)
            except Exception:
                _eprint(f"[warn] failed to read cache {cache_path.as_posix()}, reparsing")

    # Import here so the script can still print helpful messages if wandb is missing.
    from wandb.sdk.internal import datastore
    from wandb.proto.v6 import wandb_internal_pb2

    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))

    per_step: dict[int, dict[str, Any]] = {}

    while True:
        try:
            data = ds.scan_data()
        except AssertionError as e:
            if allow_incomplete:
                _eprint(f"[warn] {run.key}: stopping early due to incomplete .wandb file: {e}")
                break
            raise
        if data is None:
            break

        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)

        if rec.WhichOneof("record_type") != "history":
            continue

        step = int(rec.history.step.num)
        row = per_step.setdefault(step, {})

        for it in rec.history.item:
            if not it.key and not it.nested_key:
                continue

            # Current observed format: key is empty, nested_key stores the metric key(s).
            # For nested objects, nested_key length can be >1 (e.g. tables/artifacts).
            if it.key:
                key = it.key
                if it.nested_key:
                    key = key + "." + ".".join(it.nested_key)
            else:
                key = ".".join(it.nested_key)

            # Keep scalars only; this avoids pulling in huge table metadata blobs.
            try:
                val = _safe_json_loads(it.value_json)
            except Exception:
                continue

            if _is_number(val):
                row[key] = float(val)
            elif val is None:
                # preserve None so pandas can turn it into NaN
                row[key] = None
            else:
                # keep a few frequently useful non-numeric scalars
                if isinstance(val, str) and len(val) <= 200:
                    row[key] = val

    df = pd.DataFrame.from_dict(per_step, orient="index").sort_index()
    df.index.name = "epoch"

    if cache_path is not None:
        try:
            df.to_pickle(cache_path)
        except Exception as e:
            _eprint(f"[warn] failed to write cache {cache_path.as_posix()}: {e}")

    return df


def _metric_series(df: "pd.DataFrame", metric: str) -> tuple[list[int], list[float]]:
    if metric not in df.columns:
        return [], []
    s = df[metric]
    s = s.dropna()
    if s.empty:
        return [], []
    xs = [int(i) for i in s.index.to_list()]
    ys = [float(v) for v in s.to_list()]
    return xs, ys


def export_training_curves(
    *,
    runs: dict[str, RunInfo],
    keys: list[str],
    out_dir: Path,
    style: StyleConfig,
    cache_dir: Path | None,
    allow_incomplete: bool,
    include_compare: list[str],
    compare_label_mode: str,
    fig_w: float,
    fig_h: float,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    import pandas as pd
    from matplotlib.lines import Line2D

    out_dir.mkdir(parents=True, exist_ok=True)

    exported: list[Path] = []

    def _save_multi(fig, base: Path) -> list[Path]:
        # For split panels, always emit both PDF and PNG for convenience in thesis workflows.
        fmts = sorted(set(style.formats) | {"pdf", "png"})
        out: list[Path] = []
        for fmt in fmts:
            p = base.with_suffix(f".{fmt}")
            fig.savefig(p, bbox_inches="tight")
            out.append(p)
        return out

    for key in keys:
        if key not in runs:
            _eprint(f"[warn] training_curves: unknown key {key}, skipping")
            continue

        # Collect primary run + optional compares.
        plot_keys = [key] + [k for k in include_compare if k in runs and k != key]
        plot_runs = [runs[k] for k in plot_keys]
        dfs: dict[str, pd.DataFrame] = {}
        for r in plot_runs:
            dfs[r.key] = load_history_df(r, cache_dir=cache_dir, allow_incomplete=allow_incomplete)

        # For union MAE, color is per ablation key (A0/A1/...), while bins are distinguished via line style.
        # Make the high-occlusion bin visually most prominent.
        union_mae_metrics = [
            ("val/occl_union_mae_high", "high", "-"),
            ("val/occl_union_mae_mid", "mid", "--"),
            ("val/occl_union_mae_low", "low", ":"),
        ]

        def _plot_metric_panel(ax, *, metric: str, title: str, ylabel: str, ylim: tuple[float, float] | None) -> None:
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            if ylim is not None:
                ax.set_ylim(*ylim)

            plotted_any = False
            for rk in plot_keys:
                df = dfs[rk]
                color = style.colors.get(rk, style.colors.get("baseline", "#1f77b4"))
                xs, ys = _metric_series(df, metric)
                if not xs:
                    continue
                ys_sm = _rolling_mean(ys, style.smooth_window)
                ax.plot(
                    xs,
                    ys_sm,
                    marker="o",
                    markevery=max(1, style.marker_every),
                    color=color,
                    label=rk,
                )
                plotted_any = True

            if style.grid:
                ax.grid(True, which="both")
            if not plotted_any:
                ax.text(
                    0.5,
                    0.5,
                    "metric not logged",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color=style.colors.get("missing", "#aaaaaa"),
                )

            handles, _labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=style.legend_frameon)

        def _plot_union_mae_panel(ax) -> None:
            ax.set_title("Union prediction error")
            ax.set_xlabel("epoch")
            ax.set_ylabel("MAE")

            plotted_any = False
            for rk in plot_keys:
                df = dfs[rk]
                color = style.colors.get(rk, style.colors.get("baseline", "#1f77b4"))
                for m, _tag, ls in union_mae_metrics:
                    xs, ys = _metric_series(df, m)
                    if not xs:
                        continue
                    ys_sm = _rolling_mean(ys, style.smooth_window)
                    ax.plot(
                        xs,
                        ys_sm,
                        color=color,
                        linestyle=ls,
                        alpha=1.0 if rk == key else 0.85,
                        label="_nolegend_",
                    )
                    plotted_any = True

            if style.grid:
                ax.grid(True, which="both")
            if not plotted_any:
                ax.text(
                    0.5,
                    0.5,
                    "metric not logged",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color=style.colors.get("missing", "#aaaaaa"),
                )

            # Compact 2-part legend (color encodes run key; linestyle encodes bin).
            color_handles: list[Line2D] = []
            color_labels: list[str] = []
            for rk in plot_keys:
                c = style.colors.get(rk, style.colors.get("baseline", "#1f77b4"))
                color_handles.append(Line2D([0], [0], color=c, linestyle="-", linewidth=style.line_width))
                color_labels.append(rk)

            style_handles = [
                Line2D([0], [0], color="#444444", linestyle="-", linewidth=style.line_width),
                Line2D([0], [0], color="#444444", linestyle="--", linewidth=style.line_width),
                Line2D([0], [0], color="#444444", linestyle=":", linewidth=style.line_width),
            ]
            style_labels = ["high", "mid", "low"]

            if color_handles:
                leg1 = ax.legend(
                    color_handles,
                    color_labels,
                    loc="upper right",
                    frameon=style.legend_frameon,
                    title=None,
                )
                ax.add_artist(leg1)
            ax.legend(
                style_handles,
                style_labels,
                loc="upper right",
                bbox_to_anchor=(1.0, 0.62),
                frameon=style.legend_frameon,
                title=None,
            )

        # Combined 2x2 figure (as before).
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(fig_w, fig_h),
            constrained_layout=style.constrained_layout,
        )
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]

        _plot_metric_panel(
            ax1,
            metric="val/occl_leaf_recall_at_3",
            title="Occlusion leaf retrieval",
            ylabel="Recall@3",
            ylim=(0.0, 1.0),
        )
        _plot_metric_panel(
            ax2,
            metric="val/occl_dir_best_f1",
            title="Occlusion direction (best threshold)",
            ylabel="Occl dir F1",
            ylim=(0.0, 1.0),
        )
        _plot_metric_panel(
            ax3,
            metric="val/edge_exist_best_f1",
            title="Edge existence (best threshold)",
            ylabel="Edge exist F1",
            ylim=(0.0, 1.0),
        )
        _plot_union_mae_panel(ax4)

        for fmt in style.formats:
            out_path = out_dir / f"training_curves_{key}.{fmt}"
            fig.savefig(out_path, bbox_inches="tight")
            exported.append(out_path)
        plt.close(fig)

        # Also export split panels as individual figures in a subfolder.
        split_dir = out_dir / f"training_curves_{key}"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_w = max(3.0, fig_w / 2.0)
        split_h = max(2.3, fig_h / 2.0)

        fig1, ax = plt.subplots(1, 1, figsize=(split_w, split_h), constrained_layout=style.constrained_layout)
        _plot_metric_panel(ax, metric="val/occl_leaf_recall_at_3", title="Occlusion leaf retrieval", ylabel="Recall@3", ylim=(0.0, 1.0))
        exported += _save_multi(fig1, split_dir / "recall_at_3")
        plt.close(fig1)

        fig2, ax = plt.subplots(1, 1, figsize=(split_w, split_h), constrained_layout=style.constrained_layout)
        _plot_metric_panel(ax, metric="val/occl_dir_best_f1", title="Occlusion direction (best threshold)", ylabel="Occl dir F1", ylim=(0.0, 1.0))
        exported += _save_multi(fig2, split_dir / "occl_dir_best_f1")
        plt.close(fig2)

        fig3, ax = plt.subplots(1, 1, figsize=(split_w, split_h), constrained_layout=style.constrained_layout)
        _plot_metric_panel(ax, metric="val/edge_exist_best_f1", title="Edge existence (best threshold)", ylabel="Edge exist F1", ylim=(0.0, 1.0))
        exported += _save_multi(fig3, split_dir / "edge_exist_best_f1")
        plt.close(fig3)

        fig4, ax = plt.subplots(1, 1, figsize=(split_w, split_h), constrained_layout=style.constrained_layout)
        _plot_union_mae_panel(ax)
        exported += _save_multi(fig4, split_dir / "union_mae_bins")
        plt.close(fig4)

    return exported


def export_ablation_bars(
    *,
    runs: dict[str, RunInfo],
    out_dir: Path,
    style: StyleConfig,
    cache_dir: Path | None,
    allow_incomplete: bool,
    baseline_key: str,
    families: list[str],
    metrics: list[str],
    bar_mode: str,
    fig_w: float,
    fig_h: float,
) -> list[Path]:
    """
    Export one bar chart per family, for a small bundle of metrics.

    bar_mode:
      - "absolute": plot absolute metric values at the selected checkpoint epoch.
      - "delta": plot (run - baseline) deltas.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    import numpy as np
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)

    if baseline_key not in runs:
        raise ValueError(f"baseline_key {baseline_key} not found in discovered runs")

    # Preload history for baseline and all runs we need (keeps code simple).
    hist: dict[str, pd.DataFrame] = {}
    for k, r in runs.items():
        hist[k] = load_history_df(r, cache_dir=cache_dir, allow_incomplete=allow_incomplete)

    # Determine checkpoint epochs (best by monitor) from progress.json if present; else last epoch in history.
    selected_epoch: dict[str, int] = {}
    for k, r in runs.items():
        prog = load_progress(r.run_dir)
        if prog and isinstance(prog.get("best_monitor"), dict) and "epoch" in prog["best_monitor"]:
            selected_epoch[k] = int(prog["best_monitor"]["epoch"])
        else:
            df = hist[k]
            selected_epoch[k] = int(df.index.max()) if not df.empty else 0

    def metric_at(k: str, metric: str) -> float:
        df = hist[k]
        if df.empty or metric not in df.columns:
            return float("nan")
        ep = selected_epoch[k]
        # Prefer exact epoch; fall back to last available <= ep.
        if ep in df.index and not (pd.isna(df.loc[ep, metric])):
            v = df.loc[ep, metric]
            return float(v)
        s = df[metric].dropna()
        s = s[s.index <= ep]
        if s.empty:
            return float("nan")
        return float(s.iloc[-1])

    exported: list[Path] = []

    # Normalize bar order: baseline first, then sorted keys.
    for fam in families:
        keys = [k for k, r in runs.items() if r.family == fam]
        if not keys:
            _eprint(f"[warn] no runs found for family={fam}, skipping")
            continue
        # Ensure baseline is included for reference (even if different family).
        plot_keys = [baseline_key] + [k for k in sorted(keys) if k != baseline_key]

        # Build a metric x run matrix.
        mat = np.full((len(metrics), len(plot_keys)), np.nan, dtype=float)
        for mi, m in enumerate(metrics):
            for ki, k in enumerate(plot_keys):
                mat[mi, ki] = metric_at(k, m)

        if bar_mode == "delta":
            base = mat[:, 0:1]
            mat = mat - base

        fig, axes = plt.subplots(
            len(metrics),
            1,
            figsize=(fig_w, fig_h),
            constrained_layout=style.constrained_layout,
            sharex=True,
        )
        if len(metrics) == 1:
            axes = [axes]

        x = np.arange(len(plot_keys))
        for ax, m, y in zip(axes, metrics, mat):
            ax.set_title(m if bar_mode == "absolute" else f"{m} (delta vs {baseline_key})")
            ax.bar(
                x,
                y,
                color=[style.colors.get(k, "#777777") for k in plot_keys],
                alpha=0.9,
            )
            ax.set_ylabel("value" if bar_mode == "absolute" else "delta")
            if style.grid:
                ax.grid(True, axis="y")

            # Annotate values (small).
            for xi, yi in zip(x, y):
                if math.isnan(yi):
                    continue
                ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=style.tick_label_size)

        axes[-1].set_xticks(x, plot_keys, rotation=0)
        axes[-1].set_xlabel("run key")

        for fmt in style.formats:
            out_path = out_dir / f"ablations_{fam}_{bar_mode}.{fmt}"
            fig.savefig(out_path, bbox_inches="tight")
            exported.append(out_path)
        plt.close(fig)

    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--ablations_json",
        type=Path,
        default=Path("scripts/ablations/oracle_ablations.json"),
        help="Run registry used for key->run-name discovery.",
    )
    parser.add_argument(
        "--training_root",
        type=Path,
        default=Path("training/oracle"),
        help="Root directory containing timestamped training runs.",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default="",
        help="Comma/space-separated run keys to include (default: all found in ablations_json).",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip keys that cannot be discovered instead of failing.",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("latex_project/figures/oracle"),
        help="Output directory for exported figures.",
    )

    parser.add_argument(
        "--export",
        type=str,
        default="training_curves",
        help="Comma/space-separated exports: training_curves, ablation_bars.",
    )

    # Style overrides (coarse knobs).
    parser.add_argument("--style_json", type=Path, default=None, help="Optional JSON to override StyleConfig.")
    parser.add_argument("--formats", type=str, default="", help="Export formats (e.g. pdf,png).")
    parser.add_argument("--font_family", type=str, default=None)
    parser.add_argument("--font_size", type=float, default=None)
    parser.add_argument("--line_width", type=float, default=None)
    parser.add_argument("--smooth_window", type=int, default=None)
    parser.add_argument("--grid", type=int, default=None, help="1 enables grid, 0 disables.")

    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("analysis/wandb_history_cache"),
        help="Cache parsed histories here (pickle).",
    )
    parser.add_argument("--no_cache", action="store_true", help="Disable history cache.")
    parser.add_argument(
        "--allow_incomplete_wandb",
        action="store_true",
        help="Allow parsing partially-written .wandb files (useful if a run is still training).",
    )

    # Training curves options.
    parser.add_argument(
        "--training_curve_keys",
        type=str,
        default="A0",
        help="Keys to export training curves for.",
    )
    parser.add_argument(
        "--compare_keys",
        type=str,
        default="",
        help="Optional extra keys to overlay in the same training curve figure(s).",
    )
    parser.add_argument(
        "--compare_label_mode",
        type=str,
        default="key",
        choices=["key", "name", "key+name"],
        help="Legend labels when overlaying compare_keys.",
    )
    parser.add_argument("--train_fig_w", type=float, default=9.0)
    parser.add_argument("--train_fig_h", type=float, default=6.0)

    # Ablation bar options.
    parser.add_argument("--baseline_key", type=str, default="A0")
    parser.add_argument(
        "--families",
        type=str,
        default="loss,arch,sampling,capacity,features,objective,occl_inputs,minimal,baseline",
        help="Families to export bars for (uses family field in oracle_ablations.json).",
    )
    parser.add_argument(
        "--bar_metrics",
        type=str,
        default="val/occl_leaf_recall_at_3,val/occl_dir_best_f1,val/edge_exist_best_f1,val/occl_leaf_mass_at_3",
        help="Metrics to include in ablation bar figures (one row per metric).",
    )
    parser.add_argument(
        "--bar_mode",
        type=str,
        default="absolute",
        choices=["absolute", "delta"],
        help="Bar plot values as absolute metrics or deltas vs baseline.",
    )
    parser.add_argument("--bar_fig_w", type=float, default=9.0)
    parser.add_argument("--bar_fig_h", type=float, default=9.0)

    args = parser.parse_args()

    keys = _as_list(args.keys) or None
    export_kinds = set(_as_list(args.export))

    style = StyleConfig()
    if args.style_json is not None and args.style_json.exists():
        style.update_from_dict(json.loads(args.style_json.read_text()))
    if args.formats:
        style.formats = _as_list(args.formats)
    if args.font_family is not None:
        style.font_family = args.font_family
    if args.font_size is not None:
        style.font_size = args.font_size
        style.axes_title_size = max(style.axes_title_size, style.font_size + 1.0)
        style.axes_label_size = max(style.axes_label_size, style.font_size)
        style.tick_label_size = min(style.tick_label_size, style.font_size - 1.0)
        style.legend_font_size = min(style.legend_font_size, style.font_size - 1.0)
    if args.line_width is not None:
        style.line_width = args.line_width
    if args.smooth_window is not None:
        style.smooth_window = args.smooth_window
    if args.grid is not None:
        style.grid = bool(args.grid)

    apply_mpl_style(style)

    cache_dir = None if args.no_cache else args.cache_dir

    runs = build_run_infos(
        ablations_json=args.ablations_json,
        training_root=args.training_root,
        keys=keys,
        allow_missing=args.allow_missing,
    )

    exported: list[Path] = []
    out_dir = args.out_dir

    if "training_curves" in export_kinds:
        exported += export_training_curves(
            runs=runs,
            keys=_as_list(args.training_curve_keys),
            out_dir=out_dir,
            style=style,
            cache_dir=cache_dir,
            allow_incomplete=args.allow_incomplete_wandb,
            include_compare=_as_list(args.compare_keys),
            compare_label_mode=args.compare_label_mode,
            fig_w=args.train_fig_w,
            fig_h=args.train_fig_h,
        )

    if "ablation_bars" in export_kinds:
        exported += export_ablation_bars(
            runs=runs,
            out_dir=out_dir,
            style=style,
            cache_dir=cache_dir,
            allow_incomplete=args.allow_incomplete_wandb,
            baseline_key=args.baseline_key,
            families=_as_list(args.families),
            metrics=_as_list(args.bar_metrics),
            bar_mode=args.bar_mode,
            fig_w=args.bar_fig_w,
            fig_h=args.bar_fig_h,
        )

    if exported:
        manifest = {
            "out_dir": out_dir.as_posix(),
            "exported": [p.as_posix() for p in exported],
            "style": dataclasses.asdict(style),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Wrote {len(exported)} figure(s) to {out_dir.as_posix()}")
    else:
        print("No figures exported (check --export and --keys).")


if __name__ == "__main__":
    main()
