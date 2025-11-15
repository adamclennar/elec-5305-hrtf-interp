import pandas as pd

df = pd.read_csv("results/interp_nf_smooth/interp_metrics.csv")
df["method"] = df["method"].str.upper()
df["lsd_mean_db"] = 0.5 * (df["lsd_L_db"] + df["lsd_R_db"])

for s in sorted(df["sparsity_deg"].unique()):
    base = df[(df["sparsity_deg"] == s) & (df["method"] == "NEURAL_FIELD")]
    sm   = df[(df["sparsity_deg"] == s) & (df["method"] == "NEURAL_FIELD_SMOOTH")]

    merged = base[["subject", "lsd_mean_db"]].merge(
        sm[["subject", "lsd_mean_db"]],
        on="subject",
        suffixes=("_raw", "_smooth")
    )

    if merged.empty:
        print(f"Sparsity {s}°: no overlapping NF entries")
        continue

    merged["delta"] = merged["lsd_mean_db_smooth"] - merged["lsd_mean_db_raw"]

    print(f"\nSparsity {s}°:")
    print("Mean LSD (raw NF):    ", merged["lsd_mean_db_raw"].mean())
    print("Mean LSD (smooth NF): ", merged["lsd_mean_db_smooth"].mean())
    print("Mean Δ (smooth - raw):", merged["delta"].mean())
    print(merged["delta"].describe())
