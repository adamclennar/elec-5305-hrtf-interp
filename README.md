# HRTF Interpolation with Classical and Neural Field Methods

This repository contains code for an ELEC5305 project on **Head-Related Transfer Function (HRTF) interpolation** from sparse measurements.

It compares:

- **Nearest Neighbour (NN)**  
- **Periodic cubic spline / RBF** (classical geometric baseline)  
- **Spherical Harmonics (SH)**  
- **Residual Neural Field (NF)**  
- **Smoothed Residual Neural Field (NF-smooth)**  

All methods are evaluated under matched sparse grids (e.g. **15°** and **30°** azimuth spacing at 0° elevation) with metrics:

- Log-Spectral Distance (**LSD**) in 1–8 kHz  
- Interaural Time Difference (**ITD**) MAE  
- Interaural Level Difference (**ILD**) MAE  

The repo also includes **listening demo scripts** for A/B/C comparisons and rotating-source examples suitable for project videos.

---

## 1. Dataset: ARI HRTF SOFA Files

This project uses the **ARI HRTF database** in SOFA format.

1. Download the ARI HRTF dataset from:

   > https://projects.ari.oeaw.ac.at/research/experimental_audiology/hrtf/database/hrtfItESOFAb.html

2. Place the downloaded `.sofa` files into a folder under the repo, e.g.:

   ```text
   elec5305-project/
     data/
         NH43/
           hrtf_M_hrtf B.sofa
         NH159/
            hrtf_M_hrtf B.sofa
         ...

2. Environment Setup
2.1. Clone and create virtual environment
bash
Copy code
git clone <your-github-url> elec5305-hrtf-interp
cd elec5305-hrtf-interp

python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
2.2. Install dependencies
bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
Typical dependencies include: numpy, scipy, matplotlib, sofar, soundfile, torch, etc.
Everything needed should already be captured in requirements.txt.

3. Preprocessing: SOFA → NPZ
All experiments run on preprocessed .npz files that cache:

az_deg – azimuths (deg)

freqs_hz – frequency grid (Hz)

magL_db[A,F], magR_db[A,F] – magnitude spectra (dB)

itd_ms[A] – interaural time difference (ms)

The pipeline:

Select an elevation band (e.g. 0° ± 5°).

For each azimuth & ear:

Estimate ITD using GCC-PHAT.

Remove ITD by applying a ±ITD/2 fractional delay, so HRIRs are approximately time-aligned.

Compute minimum-phase magnitude spectra on a common frequency grid.

Save compact .npz with the fields above.

3.1. Example: convert all ARI subjects to NPZ (elev = 0°)
From the repo root:

bash
Copy code
PYTHONPATH=. python analysis/sofa_to_npz.py \
  --root data/ARI \
  --out data_npz/test \
  --elev 0
This will produce:

text
Copy code
data_npz/
  test/
    NH43__hrtf_M_hrtf_B__elev0.npz
    NH159__hrtf_M_hrtf_B__elev0.npz
    ...
You can also create separate train/ and val/ splits (for CNN / global NF training) by running sofa_to_npz.py with different --out directories or using file lists.

4. Running Interpolation Experiments
The main evaluation entry point is:

bash
Copy code
analysis/eval_methods.py
This script compares methods across subjects and sparsities, and writes metrics + plots.

4.1. Single-subject evaluation (e.g. NH43, 15° and 30°)
bash
Copy code
PYTHONPATH=. python analysis/eval_methods.py \
  --subject data_npz/test/NH43__hrtf_M_hrtf_B__elev0.npz \
  --sparsity 30 15 \
  --methods NN RBF SH NEURAL_FIELD_SMOOTH \
  --outdir results/interp_NH43
--sparsity = keep every N degrees (i.e. 30° and 15°).

--methods can include any subset of:

NN (nearest neighbour)

RBF or SPLINE (periodic cubic spline / RBF)

SH (spherical harmonics, banded)

CNN (if you have trained checkpoints)

NEURAL_FIELD / NF (per-subject residual NF)

NEURAL_FIELD_SMOOTH / NF_SMOOTH (residual NF + Savitzky–Golay smoothing)

NF_GLOBAL (if you trained a global NF model)

Outputs:

results/interp_NH43/interp_metrics.csv – per-method metrics:

lsd_L_db, lsd_R_db

lsd_L_band_db, lsd_R_band_db

itd_MAE_ms, ild_MAE_db

*.png per run – quick LSD bar plots.

4.2. All subjects under a glob
bash
Copy code
PYTHONPATH=. python analysis/eval_methods.py \
  --test_glob 'data_npz/test/*.npz' \
  --sparsity 30 15 \
  --methods NN RBF SH NEURAL_FIELD_SMOOTH \
  --outdir results/interp_ari
You can then post-process results/interp_ari/interp_metrics.csv with the provided plotting script (e.g. boxplots, mean LSD vs method) or your own analysis.

5. Neural Field Training (Optional)
There are two NF modes in this project:

Per-subject Residual Neural Field – trained on-the-fly inside predict_neural_field_residual_banded when you call eval_methods.py with NEURAL_FIELD / NEURAL_FIELD_SMOOTH.

Inputs: azimuth, baseline banded magnitudes (RBF), ear, frequency bands.

Outputs: residual corrections to the RBF magnitudes + ITD.

NF-smooth adds a Savitzky–Golay smoothing step over azimuth to enforce angular continuity.

Global NF (optional, if implemented) – trained once across many subjects, then used via NF_GLOBAL.

Example (if you have a global NF training script):

bash
Copy code
PYTHONPATH=. python analysis/train_neural_field_global.py \
  --train_glob 'data_npz/train/*.npz' \
  --out_ckpt 'results/neural_field/global_nf_residual_banded.pt'
Then evaluate with:

bash
Copy code
PYTHONPATH=. python analysis/eval_methods.py \
  --test_glob 'data_npz/test/*.npz' \
  --sparsity 30 15 \
  --methods NF_GLOBAL \
  --outdir results/interp_ari_nf_global
If you only care about per-subject NF and NF-smooth, using NEURAL_FIELD and NEURAL_FIELD_SMOOTH in eval_methods.py is enough.

6. Listening Demos
To support the project video, the repo includes two demo scripts in demo_scripts/:

make_abc_demo.py – A/B/C comparison at one azimuth

make_rotation_demo.py – rotating source demo over azimuth

6.1. Prepare demo audio sources
Place a mono WAV file for speech / noise into demo_sources/, e.g.:

text
Copy code
demo_sources/
  speech_mono.wav        # 48 kHz, mono
  pinknoise_mono.wav     # 48 kHz, mono
The scripts assume the input is 48 kHz; if not, resample beforehand.

6.2. A/B/C demo (True vs RBF vs NF-smooth)
bash
Copy code
PYTHONPATH=. python demo_scripts/make_abc_demo.py \
  --npz data_npz/test/NH43__hrtf_M_hrtf_B__elev0.npz \
  --audio_in demo_sources/speech_mono.wav \
  --az_hidden 20 \
  --step_deg 30 \
  --outdir demo_audio
This produces:

demo_audio/demo_true.wav – ground-truth HRTF at ~20°

demo_audio/demo_rbf.wav – RBF interpolation at 20°

demo_audio/demo_nf_smooth.wav – NF-smoothed interpolation at 20°

These are ideal for an A/B/C listening comparison in your video.

6.3. Rotating source demo
bash
Copy code
PYTHONPATH=. python demo_scripts/make_rotation_demo.py \
  --npz data_npz/test/NH43__hrtf_M_hrtf_B__elev0.npz \
  --audio_in demo_sources/pinknoise_mono.wav \
  --method nf_smooth \
  --step_deg 30 \
  --segment_sec 0.3 \
  --out demo_audio/rotation_nf_smooth.wav
You can also generate NN and RBF versions:

bash
Copy code
# NN rotation
PYTHONPATH=. python demo_scripts/make_rotation_demo.py \
  --npz data_npz/test/NH43__hrtf_M_hrtf_B__elev0.npz \
  --audio_in demo_sources/pinknoise_mono.wav \
  --method nn \
  --out demo_audio/rotation_nn.wav

# RBF rotation
PYTHONPATH=. python demo_scripts/make_rotation_demo.py \
  --npz data_npz/test/NH43__hrtf_M_hrtf_B__elev0.npz \
  --audio_in demo_sources/pinknoise_mono.wav \
  --method rbf \
  --step_deg 30 \
  --out demo_audio/rotation_rbf.wav
In the video, you can visualise the azimuth path (e.g. -80° → +80° → -80°) while playing these files.

7. Repository Structure (Summary)
A typical layout:

text
Copy code
elec5305-hrtf-interp/
  analysis/
    sofa_to_npz.py             # SOFA → NPZ preprocessing
    eval_methods.py            # main evaluation script (NN, RBF, SH, NF, ...)
    ... other analysis scripts (plots, boxplots, etc.)
  src/
    models/
      cnn_interp.py            # 1D CNN baseline (optional)
      ...                      # other model code
  demo_scripts/
    make_abc_demo.py           # A/B/C listening demo
    make_rotation_demo.py      # rotating source demo
  data/                        # raw SOFA (not tracked in git)
  data_npz/                    # preprocessed NPZ (not tracked in git)
  results/                     # metrics, figures, checkpoints (not tracked in git)
  demo_sources/                # mono WAVs used for demos
  requirements.txt
  README.md
  .gitignore
.gitignore should exclude large / generated artefacts:

gitignore
Copy code
data/
data_npz/
results/
*.npz
*.pt
*.wav
.venv/
__pycache__/
*.pyc
8. Reproducing the Main Results
To roughly reproduce the main comparison in the report:

Preprocess ARI subjects to NPZ (analysis/sofa_to_npz.py).

Run eval_methods.py over all test subjects with:

bash
Copy code
PYTHONPATH=. python analysis/eval_methods.py \
  --test_glob 'data_npz/test/*.npz' \
  --sparsity 30 15 \
  --methods NN RBF SH NEURAL_FIELD_SMOOTH \
  --outdir results/interp_ari
Use your plotting script (or a Jupyter notebook) to:

Compute mean LSD at 15° and 30° per method.

Make boxplots of LSD over subjects.

Inspect per-subject spectra (e.g., subject 54 at 20°).

These results underpin the report’s conclusions, e.g. that NF-smooth slightly improves mean LSD and variance over classical baselines, while NN is clearly worse.
