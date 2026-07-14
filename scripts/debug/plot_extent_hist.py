#!/usr/bin/env python3
"""Plot fruit vs stem extent distributions from a scene_graphs JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
  import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
  raise SystemExit("matplotlib is required for histogram output") from exc


def _collect_extents(graphs: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
  fruit_norms: List[float] = []
  stem_norms: List[float] = []
  fruit_radii: List[float] = []
  stem_radii: List[float] = []
  fruit_z: List[float] = []
  stem_z: List[float] = []
  for graph in graphs:
    for node in graph.get("nodes", []):
      ext = node.get("obb", {}).get("extents")
      if ext is None:
        continue
      ext_arr = np.asarray(ext, dtype=float)
      norm = float(np.linalg.norm(ext_arr))
      z_ext = float(ext_arr[2]) if ext_arr.size >= 3 else float("nan")
      ntype = node.get("type")
      if ntype == "fruit":
        fruit_norms.append(norm)
        fruit_z.append(z_ext)
        fruit_radii.append(0.5 * float(np.linalg.norm(ext_arr)))
      elif ntype == "stem":
        stem_norms.append(norm)
        stem_z.append(z_ext)
        if ext_arr.size >= 2:
          smallest = np.sort(ext_arr)[:2]
          stem_radii.append(0.5 * float(np.mean(smallest)))
  return fruit_norms, stem_norms, fruit_z, stem_z, fruit_radii, stem_radii


def main() -> None:
  parser = argparse.ArgumentParser(description="Plot fruit vs stem extent distributions.")
  parser.add_argument("--dataset-path", required=True, help="Path to scene_graphs JSON")
  parser.add_argument("--out", default=None, help="Output PNG path (default: <dataset_dir>/debug/extent_hist.png)")
  parser.add_argument("--num-scenes", type=int, default=None, help="Limit to N scenes")
  parser.add_argument("--bin-size", type=float, default=0.02, help="Histogram bin size")
  args = parser.parse_args()

  dataset_path = Path(args.dataset_path)
  graphs = json.loads(dataset_path.read_text())
  if args.num_scenes is not None:
    graphs = graphs[: max(1, args.num_scenes)]

  fruit_norms, stem_norms, fruit_z, stem_z, fruit_radii, stem_radii = _collect_extents(graphs)

  out_path = Path(args.out) if args.out else dataset_path.parent / "debug" / "extent_hist.png"
  out_path.parent.mkdir(parents=True, exist_ok=True)

  all_vals = fruit_norms + stem_norms
  if not all_vals:
    raise SystemExit("No extents found in dataset.")

  max_val = max(all_vals)
  bins = np.arange(0.0, max_val + args.bin_size, args.bin_size)

  fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  axes[0].hist(fruit_norms, bins=bins, alpha=0.7, label="fruit", color="#4c72b0")
  axes[0].hist(stem_norms, bins=bins, alpha=0.7, label="stem", color="#dd8452")
  axes[0].set_title("Extent norm (full extents)")
  axes[0].set_xlabel("||extents||")
  axes[0].set_ylabel("count")
  axes[0].legend()

  z_vals = [v for v in fruit_z + stem_z if not np.isnan(v)]
  if z_vals:
    z_max = max(z_vals)
    z_bins = np.arange(0.0, z_max + args.bin_size, args.bin_size)
    axes[1].hist(fruit_z, bins=z_bins, alpha=0.7, label="fruit z", color="#4c72b0")
    axes[1].hist(stem_z, bins=z_bins, alpha=0.7, label="stem z", color="#dd8452")
    axes[1].set_title("Z extent (height)")
    axes[1].set_xlabel("extent_z")
    axes[1].set_ylabel("count")
    axes[1].legend()
  else:
    axes[1].axis("off")

  if fruit_radii or stem_radii:
    all_r = fruit_radii + stem_radii
    r_max = max(all_r) if all_r else 1.0
    r_bins = np.arange(0.0, r_max + args.bin_size, args.bin_size)
    axes[2].hist(fruit_radii, bins=r_bins, alpha=0.7, label="fruit radius", color="#4c72b0")
    axes[2].hist(stem_radii, bins=r_bins, alpha=0.7, label="stem xy radius", color="#dd8452")
    axes[2].set_title("Stem XY radius vs Fruit radius")
    axes[2].set_xlabel("radius")
    axes[2].set_ylabel("count")
    axes[2].legend()
  else:
    axes[2].axis("off")

  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  print(f"Saved extent histogram to {out_path}")


if __name__ == "__main__":
  main()
