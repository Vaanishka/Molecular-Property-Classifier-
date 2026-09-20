# Molecular-Property-Classifier-

# Molecule Brain Barrier Predictor 🧬

> Deep learning model that predicts blood-brain barrier penetration of drug molecules using attention mechanisms.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Problem Statement

Drug development for brain diseases requires molecules that can cross the **blood-brain barrier (BBB)**. Testing BBB penetration experimentally is expensive and time-consuming. This project uses AI to predict BBB penetration from molecular structure alone.

## ✨ Features

- **Attention-based neural network** - Learns which molecular substructures matter
- **Interpretable predictions** - Visualize which atoms the model focuses on
- **Production-ready API** - Deploy as a web service
- **Resource-efficient** - Runs on CPU, optimized for small datasets
- **Docker support** - One-command deployment

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/molecule-brain-barrier.git
cd molecule-brain-barrier

# Install dependencies
pip install -r requirements.txt
```

### Predict a Molecule

```python
from src.inference import BBBPredictor

predictor = BBBPredictor.from_checkpoint('models/best_model.pt')
result = predictor.predict("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # Caffeine

print(f"Prediction: {result['class']}")           # BBB+
print(f"Confidence: {result['probability']:.2%}") # 89%
```

## 📊 Model Architecture

The model uses a **Transformer encoder** with multi-head attention:

1. **Input**: SMILES string (text representation of molecule)
2. **Tokenization**: Character-level tokens
3. **Embedding**: Convert tokens to vectors (128D)
4. **Attention Layers**: 3 encoder layers, 8 attention heads each
5. **Classification**: Binary prediction (BBB+ or BBB-)

**Why attention?** It learns which chemical groups matter (e.g., nitrogen atoms, ring structures) and we can visualize these patterns.

## 📈 Performance

Tested on BBBP dataset (2,039 molecules):

| Metric | Score |
|--------|-------|
| Accuracy | 89.2% |
| ROC-AUC | 0.94 |
| F1 Score | 0.88 |

## 🧪 Dataset

Uses the **BBBP (Blood-Brain Barrier Penetration)** dataset from [MoleculeNet](https://moleculenet.org/):
- 2,039 molecules
- Binary labels (BBB+ / BBB-)
- Automatically downloaded during training

## 🏗️ Project Structure
