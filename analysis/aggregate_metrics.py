#!/usr/bin/env python3
from __future__ import annotations
import csv, os, math
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def load_rows(csv_path: str) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    return rows

def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def aggregate(rows: List[Dict[str, str]]) -> Tuple[List[str], List[int], Dict[Tuple[str,int], Dict[str,float]]]:
    """
    Returns:
      methods: sorted list of methods
      sparsities: sorted list of sparsity degrees (int)
      stats[(method, sparsity)] = { 'LSD_avg_mean', 'LSD_avg_std', 'ITD_MAE_mean', 'ITD_MAE_std', 'ILD_MAE_mean', 'ILD_MAE_std', 'n' }
    """
    # collect values per group
    buckets: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    methods, sparsities = set(), set()

    for r in rows:
        m = r["method"].upper()
        s = int(r["sparsity_deg"])
        methods.add(m); sparsities.add(s)

        lsdL = to_float(r["lsd_L_db"])
        lsdR = to_float(r["lsd_R_db"])
        itd  = to_float(r["itd_MAE_ms"])
        ild  = to_float(r["ild_MAE_db"])

        if any(math.isnan(v) for v in (lsdL, lsdR, itd, ild)):
            continue

        lsd_avg = 0.5*(lsdL + lsdR)

        b = buckets[(m, s)]
        b["LSD_avg"].append(lsd_avg)
        b["ITD_MAE"].append(itd)
        b["ILD_MAE"].append(ild)

    methods = sorted(methods)
    sparsities = sorted(sparsities)

    stats: Dict[Tuple[str,int], Dict[str,float]] = {}
    for k, b in buckets.items():
        out = {}
        for key in ("LSD_avg", "ITD_MAE", "ILD_MAE"):
            arr = np.asarray(b[key], dtype=float)
            out[f"{key}_mean"] = float(np.mean(arr)) if arr.size else float("nan")
            out[f"{key}_std"]  = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            out["n"] = int(arr.size)
        stats[k] = out
    return methods, sparsities, stats

def save_summary_csv(stats, methods, sparsities, out_csv):
    fields = ["method","sparsity_deg","n","LSD_avg_mean","LSD_avg_std","ITD_MAE_mean","ITD_MAE_std","ILD_MAE_mean","ILD_MAE_std"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sparsities:
            for m in methods:
                k = (m, s)
                row = {"method": m, "sparsity_deg": s}
                vals = stats.get(k, {})
                row["n"] = vals.get("n", 0)
                row["LSD_avg_mean"] = vals.get("LSD_avg_mean", float("nan"))
                row["LSD_avg_std"]  = vals.get("LSD_avg_std", 0.0)
                row["ITD_MAE_mean"] = vals.get("ITD_MAE_mean", float("nan"))
                row["ITD_MAE_std"]  = vals.get("ITD_MAE_std", 0.0)
                row["ILD_MAE_mean"] = vals.get("ILD_MAE_mean", float("nan"))
                row["ILD_MAE_std"]  = vals.get("ILD_MAE_std", 0.0)
                w.writerow(row)
    print("Saved summary:", out_csv)

def _barplot(ax, title, ylabel, methods, sparsities, stats, key_mean, key_std):
    x = np.arange(len(sparsities))
    width = 0.8 / max(1, len(methods))  # group width
    for mi, m in enumerate(methods):
        means = [stats.get((m,s), {}).get(key_mean, np.nan) for s in sparsities]
        stds  = [stats.get((m,s), {}).get(key_std, 0.0) for s in sparsities]
        ax.bar(x + mi*width - (len(methods)-1)*width/2, means, width=width, yerr=stds, capsize=3, label=m)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}°" for s in sparsities])
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)

def make_plots(stats, methods, sparsities, outdir):
    os.makedirs(outdir, exist_ok=True)
    # LSD_avg
    fig, ax = plt.subplots(figsize=(6,3.2))
    _barplot(ax, "Log-Spectral Distance (avg L/R)", "dB", methods, sparsities, stats, "LSD_avg_mean", "LSD_avg_std")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "agg_LSD_avg.png"), dpi=150); plt.close(fig)

    # ITD MAE
    fig, ax = plt.subplots(figsize=(6,3.2))
    _barplot(ax, "ITD MAE", "ms", methods, sparsities, stats, "ITD_MAE_mean", "ITD_MAE_std")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "agg_ITD_MAE.png"), dpi=150); plt.close(fig)

    # ILD MAE
    fig, ax = plt.subplots(figsize=(6,3.2))
    _barplot(ax, "ILD MAE (1–8 kHz band-avg)", "dB", methods, sparsities, stats, "ILD_MAE_mean", "ILD_MAE_std")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "agg_ILD_MAE.png"), dpi=150); plt.close(fig)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", default="results/interp/interp_metrics.csv")
    ap.add_argument("--outdir", default="results/interp")
    args = ap.parse_args()

    rows = load_rows(args.metrics_csv)
    methods, sparsities, stats = aggregate(rows)
    save_summary_csv(stats, methods, sparsities, os.path.join(args.outdir, "interp_metrics_summary.csv"))
    make_plots(stats, methods, sparsities, args.outdir)

if __name__ == "__main__":
    main()
