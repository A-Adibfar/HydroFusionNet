# HydroFusionNet - Precipitation Classifier

"HydroFusionNet is a multi-stream self-attention fusion network designed for precipitation classification. It separately processes spatial/temporal feature sets, fuses them via adaptive attention, integrates physics-informed differential features, and embeds seasonal patterns using cyclic temporal encodings, achieving robust performance on noisy meteorological data.

## Structure

- `models/` — Model architecture (MSAFNet)
- `datasets/` — Custom dataset class
- `training/` — Training and evaluation functions
- `utils/` — Preprocessing and data utilities
- `classifierResults/` — Saved evaluation results (confusion matrix, ROC curve, etc.)

---

## Setup

```bash
git clone https://github.com/yourname/precipitation_classifier.git
cd precipitation_classifier
pip install -r requirements.txt
