ELEC5305 — HRTF Interpolation for Sparse Spatial Mixing

This repo explores how well different interpolation strategies reconstruct missing HRTF directions from sparse measurements. We compare a simple Nearest-Neighbour (NN) baseline against a periodic spline (smooth circular interpolation). Code produces quantitative metrics and QC plots and is structured so a CNN (& possibly spherical CNN) model can be used later.

Project aim

Reconstruct missing HRTF directions from sparse measurements and compare NN vs a smooth periodic spline (and later a CNN) using:

Spectral error: Log-Spectral Distance (LSD) per ear (dB)

Timing error: ITD MAE (ms)

Level error: band-averaged ILD MAE (dB)

(Planned) Motion smoothness during panning

What’s done so far

Implemented preprocessing:

Load SOFA HRIRs; select an elevation band (e.g., 0°±5°)

Estimate ITD (phase-slope primary, GCC fallback; consistent sign)

Time-align ears (remove ±ITD/2 via fractional delay)

Compute per-ear spectra (dB) on fixed frequency grid

Save compact features to .npz for fast training/eval

QC scripts:

ITD vs azimuth, magnitude heatmaps, ILD vs azimuth, angle coverage

Baselines:

NN and Periodic spline interpolation along azimuth

Metrics: LSD (dB RMSE), ITD MAE (ms), ILD MAE (dB)

Aggregation:

Mean ± SD across subjects and sparsities, with plots

Repository layout
analysis/
  qc_plots.py            # ITD/ILD/heatmaps/coverage for a single .npz
  check_itd_vs_az.py     # (optional) focused ITD vs azimuth plot
  eval_methods.py        # run NN / spline baselines + save metrics
  aggregate_metrics.py   # aggregate metrics across subjects
data/                    # (you put .sofa files here, e.g., data/NH43/*.sofa)
data_npz/                # preprocessed feature files (.npz)
results/                 # metrics CSVs and figures
src/
  sofa_loader.py         # SOFA reader + elevation selection
  signal_tools.py        # ITD, frac-delay, spectra, helpers
  data_prep.py           # SOFA -> .npz (single + bulk)
splits/
  train.txt, val.txt, test.txt   # newline-separated paths to .sofa files
README.md
requirements.txt


Naming convention for outputs (no subfolders):
data_npz/test/NH43__hrtf_M_hrtf B__elev0.npz
(subject inferred from the parent directory name like NH43)

Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


Place SOFA files under subject-named folders, e.g.

data/NH43/hrtf_M_hrtf B.sofa
data/NH5/hrtf_M_hrtf B.sofa
...


Then list them (absolute or relative paths) in splits/train.txt, splits/val.txt, splits/test.txt — one path per line.

Reproduce

End-to-end: preprocess → QC → evaluate baselines → aggregate results.

# 0) Activate env
source .venv/bin/activate

# 1) Preprocess SOFA -> NPZ (elevation 0° ± 5°)
PYTHONPATH=. python - <<'PY'
from src.data_prep import bulk_sofa_to_npz
bulk_sofa_to_npz('splits/train.txt','data_npz/train', elev=0, tol=5, nfft=2048)
bulk_sofa_to_npz('splits/val.txt',  'data_npz/val',   elev=0, tol=5, nfft=2048)
bulk_sofa_to_npz('splits/test.txt', 'data_npz/test',  elev=0, tol=5, nfft=2048)
PY

# 2) QC a few subjects (adjust filename)
PYTHONPATH=. python analysis/qc_plots.py \
  --npz data_npz/test/NH43__hrtf_M_hrtf B__elev0.npz \
  --outdir results/qc

# (Optional) focused ITD vs azimuth
PYTHONPATH=. python analysis/check_itd_vs_az.py \
  --npz data_npz/test/NH43__hrtf_M_hrtf B__elev0.npz \
  --out results/qc/NH43_itd_vs_az.png

# 3) Evaluate baselines (NN and periodic spline) at multiple sparsities
PYTHONPATH=. python analysis/eval_methods.py \
  --test_glob 'data_npz/test/*.npz' \
  --sparsity 30 15 10 \
  --methods NN RBF \
  --outdir results/interp

# 4) Aggregate metrics across subjects
PYTHONPATH=. python analysis/aggregate_metrics.py \
  --metrics_csv results/interp/interp_metrics.csv \
  --outdir results/interp


Outputs you should see

results/qc/*_itd_vs_az.png, *_mag_heatmap_*.png, *_ild_vs_az.png, *_angle_coverage.png

results/interp/interp_metrics.csv (per-subject rows)

results/interp/interp_metrics_summary.csv (mean ± SD by method/sparsity)

results/interp/agg_LSD_avg.png, agg_ITD_MAE.png, agg_ILD_MAE.png

Notes & assumptions

ITD sign convention: positive = Right ear later than Left. The pipeline enforces this; if you bring external NPZs, keep it consistent.

ILD band: 1–8 kHz band-average (adjust in scripts if needed).

Spline method: “RBF” flag maps to a periodic cubic spline across azimuth (robust, no hyper-params). True RBF can be added later.

Elevation slice: default elev=0, tol=5. Tighten to tol=3 if needed.

Roadmap

Add CNN interpolator (input = sparse azimuths, output = dense mags + ITD).

Add motion smoothness metric + render A/B panning WAVs.

Compare across sparsities and subjects; ablate ITD clean-up options.

Data policy / versioning

Keep the repo lean. Large artifacts (*.sofa, *.npz, *.wav) are ignored by default.
If you must include sample .npz, use Git LFS:

git lfs install
git lfs track "*.npz" "*.sofa" "*.wav"
git add .gitattributes
