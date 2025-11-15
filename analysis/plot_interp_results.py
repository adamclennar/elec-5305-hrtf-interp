#!/usr/bin/env python3
"""
Plot comparison of interpolation methods from interp_metrics.csv.

Generates:
  1) mean_LSD_bar_[sparsity].png
  2) lsd_box_[sparsity].png
  3) nf_minus_rbf_delta_hist_[sparsity].png   (if both NF & RBF exist)
  4) nf_vs_rbf_scatter_[sparsity].png         (if both NF & RBF exist)

Assumes CSV columns include at least:
  subject, method, sparsity_deg,
  lsd_L_db, lsd_R_db, itd_MAE_ms, ild_MAE_db
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Combined LSD (mean of L/R)
    if "lsd_mean_db" not in df.columns:
        df["lsd_mean_db"] = 0.5 * (df["lsd_L_db"] + df["lsd_R_db"])
    # Normalise method names a bit
    df["method"] = df["method"].astype(str).str.upper()
    return df


def bar_mean_lsd(df: pd.DataFrame, outdir: str):
    sparsities = sorted(df["sparsity_deg"].unique())
    for s in sparsities:
        sub = df[df["sparsity_deg"] == s]
        if sub.empty:
            continue

        g = (sub.groupby("method")["lsd_mean_db"]
                 .mean()
                 .sort_values())
        methods = g.index.tolist()
        vals = g.values

        plt.figure(figsize=(6, 4))
        x = range(len(methods))
        plt.bar(x, vals)
        plt.xticks(x, methods, rotation=30, ha="right")
        plt.ylabel("Mean LSD (dB)")
        plt.title(f"Mean LSD across subjects @ {s}° sparsity")
        plt.tight_layout()
        fname = os.path.join(outdir, f"mean_LSD_bar_{s}deg.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"[saved] {fname}")

def bar_mean_lsd_zoom(df: pd.DataFrame, outdir: str):
    """
    Zoomed bar plots for advanced methods only (exclude NN),
    with tight y-limits to reveal small differences.
    """
    sparsities = sorted(df["sparsity_deg"].unique())
    for s in sparsities:
        sub = df[df["sparsity_deg"] == s]
        if sub.empty:
            continue

        # keep only "good" methods
        keep_methods = ["RBF", "SH", "NEURAL_FIELD", "NEURAL_FIELD_SMOOTH", "NF", "NF_SMOOTH", "NF_GLOBAL"]
        sub = sub[sub["method"].isin(keep_methods)]
        if sub.empty:
            continue

        g = (sub.groupby("method")["lsd_mean_db"]
                 .mean()
                 .sort_values())
        methods = g.index.tolist()
        vals = g.values

        # compute tight y-limits
        vmin = vals.min()
        vmax = vals.max()
        margin = (vmax - vmin) * 0.5 if vmax > vmin else 0.05
        ymin = vmin - margin
        ymax = vmax + margin

        plt.figure(figsize=(6, 4))
        x = range(len(methods))
        plt.bar(x, vals)
        plt.xticks(x, methods, rotation=30, ha="right")
        plt.ylabel("Mean LSD (dB)")
        plt.title(f"Mean LSD (zoomed) @ {s}° (advanced methods)")
        plt.ylim(ymin, ymax)
        plt.tight_layout()
        fname = os.path.join(outdir, f"mean_LSD_bar_zoom_{s}deg.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"[saved] {fname}")


def boxplot_lsd(df: pd.DataFrame, outdir: str):
    sparsities = sorted(df["sparsity_deg"].unique())
    for s in sparsities:
        sub = df[df["sparsity_deg"] == s]
        if sub.empty:
            continue

        # Fix method order: NN, RBF, SH, NEURAL_FIELD, NF_GLOBAL if present
        methods_all = sorted(sub["method"].unique())
        preferred = ["NN", "RBF", "SPLINE", "SH", "CNN", "NEURAL_FIELD", "NF", "NF_GLOBAL"]
        methods = [m for m in preferred if m in methods_all]
        methods += [m for m in methods_all if m not in methods]

        data = [sub[sub["method"] == m]["lsd_mean_db"].values for m in methods]

        plt.figure(figsize=(7, 4))
        plt.boxplot(data, tick_labels=methods, showfliers=True)
        plt.ylabel("LSD (dB)")
        plt.title(f"LSD distribution across subjects @ {s}° sparsity")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fname = os.path.join(outdir, f"lsd_box_{s}deg.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"[saved] {fname}")

def boxplot_lsd_zoom(df: pd.DataFrame, outdir: str):
    sparsities = sorted(df["sparsity_deg"].unique())
    for s in sparsities:
        sub = df[df["sparsity_deg"] == s]
        if sub.empty:
            continue

        keep_methods = ["RBF", "SH", "NEURAL_FIELD", "NEURAL_FIELD_SMOOTH", "NF", "NF_SMOOTH", "NF_GLOBAL"]
        sub = sub[sub["method"].isin(keep_methods)]
        if sub.empty:
            continue

        methods_all = sorted(sub["method"].unique())
        preferred = ["RBF", "SH", "NEURAL_FIELD", "NEURAL_FIELD_SMOOTH", "NF", "NF_SMOOTH", "NF_GLOBAL"]
        methods = [m for m in preferred if m in methods_all]

        data = [sub[sub["method"] == m]["lsd_mean_db"].values for m in methods]

        # tight y limits over all values
        all_vals = np.concatenate(data)
        vmin = all_vals.min()
        vmax = all_vals.max()
        margin = (vmax - vmin) * 0.5 if vmax > vmin else 0.05
        ymin = vmin - margin
        ymax = vmax + margin

        plt.figure(figsize=(7, 4))
        plt.boxplot(data, labels=methods, showfliers=True)
        plt.ylabel("LSD (dB)")
        plt.title(f"LSD distribution (zoomed) @ {s}° (advanced methods)")
        plt.ylim(ymin, ymax)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fname = os.path.join(outdir, f"lsd_box_zoom_{s}deg.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"[saved] {fname}")


def nf_vs_rbf_delta_plots(df: pd.DataFrame, outdir: str, nf_label="NEURAL_FIELD"):
    """
    Makes:
      - histogram of (LSD_nf - LSD_rbf)
      - scatter of LSD_rbf vs LSD_nf
    for each sparsity where both exist.
    """
    sparsities = sorted(df["sparsity_deg"].unique())
    for s in sparsities:
        sub = df[df["sparsity_deg"] == s]
        if sub.empty:
            continue

        # Handle cases where NF is recorded as "NF" instead of "NEURAL_FIELD"
        methods_here = set(sub["method"].unique())
        if nf_label not in methods_here and "NF" in methods_here:
            nf_label_use = "NF"
        else:
            nf_label_use = nf_label

        if "RBF" not in methods_here or nf_label_use not in methods_here:
            print(f"[info] Skipping NF vs RBF plots @ {s}° (missing RBF or {nf_label_use})")
            continue

        base = sub[sub["method"] == "RBF"][["subject", "lsd_mean_db"]]
        nf = sub[sub["method"] == nf_label_use][["subject", "lsd_mean_db"]]

        merged = base.merge(nf, on="subject", suffixes=("_rbf", "_nf"))
        if merged.empty:
            print(f"[info] No overlapping subjects for RBF and {nf_label_use} at {s}°")
            continue

        merged["delta_nf_minus_rbf"] = merged["lsd_mean_db_nf"] - merged["lsd_mean_db_rbf"]

        # Histogram of deltas
        plt.figure(figsize=(6, 4))
        plt.hist(merged["delta_nf_minus_rbf"], bins=20)
        plt.axvline(0.0, linestyle="--")
        plt.xlabel("LSD(NF) - LSD(RBF)  [dB]")
        plt.ylabel("Count")
        plt.title(f"{nf_label_use} vs RBF LSD difference @ {s}° (neg = NF better)")
        plt.tight_layout()
        fname = os.path.join(outdir, f"nf_minus_rbf_delta_hist_{s}deg.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"[saved] {fname}")

        # Scatter RBF vs NF
        plt.figure(figsize=(5, 5))
        plt.scatter(merged["lsd_mean_db_rbf"], merged["lsd_mean_db_nf"])
        lim_min = min(merged["lsd_mean_db_rbf"].min(), merged["lsd_mean_db_nf"].min()) - 0.2
        lim_max = max(merged["lsd_mean_db_rbf"].max(), merged["lsd_mean_db_nf"].max()) + 0.2
        plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.xlabel("RBF LSD (dB)")
        plt.ylabel(f"{nf_label_use} LSD (dB)")
        plt.title(f"{nf_label_use} vs RBF per subject @ {s}°")
        plt.tight_layout()
        fname = os.path.join(outdir, f"nf_vs_rbf_scatter_{s}deg.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"[saved] {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to interp_metrics.csv")
    ap.add_argument("--outdir", required=True, help="Directory for output plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_df(args.csv)


    bar_mean_lsd(df, args.outdir)
    boxplot_lsd(df, args.outdir)
    nf_vs_rbf_delta_plots(df, args.outdir, nf_label="NEURAL_FIELD")
    # NEW:
    bar_mean_lsd_zoom(df, args.outdir)
    boxplot_lsd_zoom(df, args.outdir)


if __name__ == "__main__":
    main()
