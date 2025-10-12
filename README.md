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

