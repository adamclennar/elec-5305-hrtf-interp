# elec5305-project-520455209

Evaluating the Localisation Accuracy of HRIR Datasets for Headphone Spatialisation

Abstract:

Accurate spatial audio reproduction through headphones depends heavily on the choice of Head-Related Impulse Response (HRIR) dataset used for binaural rendering. This project develops a Python-based binaural spatialisation tool that allows direct comparison between commonly used HRIR datasets such as CIPIC, KEMAR, and ARI. The tool performs HRIR-based convolution of mono audio sources, computes interaural time (ITD) and level differences (ILD) as a function of azimuth, and facilitates short listening experiments to evaluate localisation accuracy. Both objective (ITD/ILD linearity and smoothness) and subjective (mean absolute azimuth error and externalisation ratings) metrics are used to assess how dataset choice affects spatial fidelity in headphone music production. The outcomes will inform which datasets yield the most natural and consistent spatial imaging, contributing to improved headphone-based mixing workflows.

Goals:

	1. Implement an offline HRIR spatialiser that loads SOFA-format datasets and renders binaural audio from mono stems.
	
	2. Quantify interaural time and level differences (ITD/ILD) versus azimuth for multiple HRIR datasets.
	
	3. Conduct subjective listening tests to estimate perceived localisation error and externalisation quality.
	
	4. Compare and visualise objective and subjective results to identify datasets offering superior spatial accuracy for headphone playback.
	
	5. Publish all code, analysis scripts, and result plots in an open GitHub repository for reproducibility.
	


# Evaluating HRIR Datasets for Headphone Spatialisation

This project develops a Python-based toolchain to (1) render mono audio to binaural via HRIR convolution (SOFA files) and (2) analyse **interaural time difference (ITD)** and **interaural level difference (ILD)** versus azimuth for different datasets.

### What’s here now
- **Renderer:** `src/render.py` — mono→binaural (FFT convolution), simple distance model.
- **Analysis:** `analysis/itd_ild_analysis.py` — ITD/ILD plots + dataset alignment + Δ(ITD/ILD) with MAE/RMSE.
- **Preliminary results:** KEMAR 5° vs 0.5° (upsampled) at 0° elevation — curves overlap at matched angles, as expected.

### Install
```bash
python -m venv .venv
source .venv/bin/activate    
pip install -r requirements.txt


# handy commands !!
# Preprocess
PYTHONPATH=. python -c "from src.data_prep import bulk_sofa_to_npz; \
bulk_sofa_to_npz('splits/test.txt','data_npz/test')"

# Baselines evaluation (start with one subject)
PYTHONPATH=. python analysis/eval_methods.py \
  --subject data_npz/test/subj_000.npz \
  --sparsity 30 \
  --methods NN RBF \
  --outdir results/interp

# CNN training (after baselines)
PYTHONPATH=. python analysis/train_cnn.py \
  --train_glob 'data_npz/train/*.npz' \
  --val_glob   'data_npz/val/*.npz' \
  --epochs 30 --batch 16 --lr 1e-3 --tv 1e-3 \
  --save results/cnn_ckpt.pt

# Full evaluation
PYTHONPATH=. python analysis/eval_methods.py \
  --test_glob 'data_npz/test/*.npz' \
  --sparsity 30 15 10 \
  --methods NN RBF CNN \
  --cnn_ckpt results/cnn_ckpt.pt \
  --outdir results/interp
