# EASE - EEG Analysis for Seizure Evaluation

A deep learning framework for automated seizure detection from EEG signals using CNNs, transfer learning, and explainable AI.

## Project Structure

```
EASE/
├── 1d model/              # 1D CNN for raw EEG signals
├── spectrograms/          # 2D CNN for spectrogram-based detection
├── TRANSFER/              # Transfer learning (EfficientNet, ResNet, MobileNet)
├── LLM_TL code/           # XAI + LLM integration for explainability
├── xai_llm/               # XAI output reports and visualizations
├── chbmit/                # CHB-MIT dataset utilities
└── spectrogramsnpy.py     # Spectrogram generation script
```

## Models

| Model | Architecture | Input | Description |
|-------|-------------|-------|-------------|
| 1D CNN | EEGWaveNet-style | Raw EEG | Temporal pattern detection |
| 2D CNN | Custom 4-layer | Spectrograms | Frequency-based detection |
| EfficientNet-B0 + LSTM | Transfer learning | Spectrograms | State-of-the-art performance |
| ResNet-34 | Transfer learning | Spectrograms | Deep feature extraction |
| MobileNet | Transfer learning | Spectrograms | Lightweight deployment |

## Features

- **Multi-architecture support**: 1D CNNs, 2D CNNs, and transfer learning models
- **Patient-specific training**: Individual models per patient for personalized detection
- **Explainable AI (XAI)**: GradCAM, GradCAM++, and Integrated Gradients visualizations
- **LLM-powered reports**: Automated clinical interpretation of model predictions

## Installation

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib mne
pip install efficientnet_pytorch  # for transfer learning models
```

## Usage

### Generate Spectrograms
```bash
python spectrogramsnpy.py
```

### Train 2D CNN
```bash
python spectrograms/2dcnns/train_all_patients.py
```

### Train with Transfer Learning
```bash
python TRANSFER/TLcode/train_eff.py  # EfficientNet
python TRANSFER/TLcode/train_res.py  # ResNet
python TRANSFER/TLcode/train_mob.py  # MobileNet
```

### Generate XAI Reports
```bash
python LLM_TL\ code/xai_r.py
```

## Dataset

Tested on:
- **CHB-MIT Scalp EEG Database** - Pediatric seizure recordings
