ELEC5305 — HRTF Interpolation from Sparse Directions

This project evaluates how well different interpolation strategies reconstruct missing HRTF directions for headphone spatial mixing from sparse azimuth samples on the horizontal plane (elev ≈ 0°). We currently compare a simple Nearest-Neighbour (NN) baseline versus a smooth periodic spline (circular cubic spline). A CNN model will be added later.

Goals

Reconstruct dense azimuth HRTFs from sparse measurements.

Compare methods using:

LSD (Log-Spectral Distance) per ear (dB, band-limited),

ITD MAE (ms),

ILD MAE (dB, band-average).

(Planned) Evaluate motion smoothness during panning and provide A/B demo audio.

What’s implemented

Preprocessing

Load SOFA, select elevation band (default: 0° ± 5°).

Estimate ITD (phase-slope primary, GCC-PHAT fallback), consistent sign: + = Right later than Left.

Time-align ears (remove ±ITD/2 with fractional delay).

Compute fixed-grid magnitude spectra (dB) per ear.

Save compact features to .npz for fast experiments.

QC & visualisation: ITD vs azimuth, ILD vs azimuth, magnitude heatmaps, angle coverage.

Baselines & metrics

NN along azimuth (circular).

Periodic spline (circular cubic spline) for per-freq magnitudes and for ITD.

Metrics: LSD_L, LSD_R, ITD_MAE, ILD_MAE; CSV + figures.

Aggregation: mean ± SD across subjects and sparsities; summary plots.

Repository layout
analysis/
  qc_plots.py            # QC for a single .npz (ITD/ILD/heatmaps/coverage)
  check_itd_vs_az.py     # Focused ITD vs azimuth plot
  eval_methods.py        # Run NN / spline baselines + metrics
  aggregate_metrics.py   # Aggregate metrics across subjects
results/
  interp/                # metrics CSV + aggregate PNGs
  qc/                    # subject QC PNGs
splits/
  train.txt  val.txt  test.txt   # newline-separated paths to .sofa files
src/
  sofa_loader.py         # SOFA I/O + elevation selection
  signal_tools.py        # ITD, fractional delay, spectra, helpers
  data_prep.py           # SOFA → .npz (single + bulk)
README.md
requirements.txt


Not tracked (see .gitignore): data/ (SOFA), data_npz/ (features), out/, audio WAVs, and any large artifacts.

Reproduce

End-to-end: preprocess → QC → evaluate baselines → aggregate results.
Place your SOFA files under data/<SUBJECT>/… (e.g., data/NH43/hrtf_M_hrtf B.sofa) and list them in splits/*.txt.

# 0) Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Preprocess SOFA → NPZ (elev 0° ± 5°)
PYTHONPATH=. python - <<'PY'
from src.data_prep import bulk_sofa_to_npz
bulk_sofa_to_npz('splits/train.txt','data_npz/train', elev=0, tol=5, nfft=2048)
bulk_sofa_to_npz('splits/val.txt',  'data_npz/val',   elev=0, tol=5, nfft=2048)
bulk_sofa_to_npz('splits/test.txt', 'data_npz/test',  elev=0, tol=5, nfft=2048)
PY

# 2) QC (pick one subject NPZ)
PYTHONPATH=. python analysis/qc_plots.py \
  --npz data_npz/test/<SUBJECT>__<sofa-base>__elev0.npz \
  --outdir results/qc

# (Optional) Focused ITD vs azimuth
PYTHONPATH=. python analysis/check_itd_vs_az.py \
  --npz data_npz/test/<SUBJECT>__<sofa-base>__elev0.npz \
  --out results/qc/<SUBJECT>_itd_vs_az.png

# 3) Evaluate baselines at multiple sparsities (e.g., 30°, 15°, 10°)
PYTHONPATH=. python analysis/eval_methods.py \
  --test_glob 'data_npz/test/*.npz' \
  --sparsity 30 15 10 \
  --methods NN RBF \
  --outdir results/interp

# 4) Aggregate metrics across subjects
PYTHONPATH=. python analysis/aggregate_metrics.py \
  --metrics_csv results/interp/interp_metrics.csv \
  --outdir results/interp


Outputs

Per-subject QC PNGs under results/qc/.

Per-subject metrics rows in results/interp/interp_metrics.csv.

Aggregate summary CSV + figures:

results/interp/interp_metrics_summary.csv

results/interp/agg_LSD_avg.png

results/interp/agg_ITD_MAE.png

results/interp/agg_ILD_MAE.png

Current snapshot (example)

On NH43 at 30° sampling, periodic spline vs NN:

LSD(avg L/R): ↓ ~17–20%

ITD MAE: 0.067 → 0.010 ms

ILD MAE: 1.41 → 1.05 dB

(See results/interp/*.png for the exact figures.)

Notes & assumptions

ITD sign convention: positive = Right ear later than Left (enforced in preprocessing).

ILD band: default 1–8 kHz band-average for ILD plots/metrics.

Angle wrap: azimuths are wrapped to [−180, 180) and sorted for interpolation.

Periodic spline: “RBF” flag maps to a circular cubic spline; true RBF can be added later.
