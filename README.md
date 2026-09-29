Here's a comprehensive GitHub README for your project (without Docker):

---

# 🧬 Blood-Brain Barrier Penetration Predictor

A deep learning system for predicting whether small molecules can cross the blood-brain barrier (BBB) using transformer-based neural networks.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

## 🎯 Overview

Predicting blood-brain barrier penetration is crucial for developing drugs targeting neurological diseases. This project uses a custom transformer architecture to classify molecules based on their SMILES representations, achieving **87% accuracy** on the test set.

**Key Capabilities:**
- Predict BBB penetration for single molecules
- Batch processing for large compound libraries
- REST API for integration with other tools
- Interactive web interface
- Rule-based validation (Lipinski's rules)

---

## ✨ Features

- **Transformer-Based Architecture**: Attention mechanisms capture long-range dependencies in molecular structures
- **SMILES Tokenization**: Custom tokenizer for molecular representations
- **Production-Ready API**: FastAPI backend with automatic documentation
- **Interactive Frontend**: Web interface for real-time predictions
- **Batch Inference**: Process thousands of molecules efficiently
- **Comprehensive Logging**: Track training progress and model performance


## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/molecular_bb.git
cd molecular_bb
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the trained model**

Place your trained model file at:
```
models/bbb_model.pt
```

---

## 💻 Usage

### 1. Command-Line Inference

**Single molecule prediction:**
```bash
python -m src.inference --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
```

**Batch prediction from CSV:**
```bash
python -m src.inference --input_csv molecules.csv --output_csv predictions.csv
```

**Interactive mode:**
```bash
python -m src.inference
```

---

### 2. Web Interface

**Start the API server:**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Access the app:**
- Frontend: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

### 3. REST API

**Single prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}'
```

**Batch prediction:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "molecules": [
      {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
      {"smiles": "CCO"}
    ]
  }'
```

---

### 4. Python Script

```python
from src.inference import BBBPredictor

# Initialize predictor
predictor = BBBPredictor(checkpoint_path='models/bbb_model.pt')

# Predict single molecule
result = predictor.predict_single('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
print(f"Prediction: {result['predicted_class']}")
print(f"Probability: {result['bbb_positive_probability']:.4f}")

# Batch prediction
smiles_list = ['CCO', 'c1ccccc1', 'CC(C)Cc1ccc(cc1)C(C)C(O)=O']
results = predictor.predict_batch(smiles_list)
```

---

## 📁 Project Structure

```
molecular_bb/
├── api/                          # FastAPI application
│   ├── __init__.py
│   ├── main.py                   # API endpoints
│   ├── schemas.py                # Request/response models
│   ├── inference_service.py      # Model serving logic
│   └── static/
│       └── index.html            # Web frontend
│
├── src/                          # Source code
│   ├── data/
│   │   ├── dataset.py            # PyTorch dataset
│   │   └── __init__.py
│   ├── models/
│   │   ├── smilesx_model.py      # Transformer architecture
│   │   └── __init__.py
│   ├── utils/
│   │   ├── smiles_tokenizer.py   # SMILES tokenization
│   │   ├── featurizer.py         # Molecular features
│   │   └── __init__.py
│   ├── training/
│   │   ├── train.py              # Training script
│   │   ├── evaluate.py           # Evaluation metrics
│   │   └── __init__.py
│   ├── inference.py              # CLI inference
│   └── __init__.py
│
├── config/                       # Configuration files
│   └── config.yaml
│
├── data/                         # Datasets
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/                       # Saved models
│   └── bbb_model.pt
│
├── notebooks/                    # Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
├── results/                      # Training results
│   ├── plots/
│   └── logs/
│
├── tests/                        # Unit tests
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🧠 Model Architecture

### Transformer Encoder

```
Input (SMILES) → Tokenization → Embedding (256d)
                                    ↓
                            Positional Encoding
                                    ↓
                    ┌─────────────────────────────┐
                    │   Transformer Encoder       │
                    │   - 4 layers                │
                    │   - 8 attention heads       │
                    │   - 512 hidden dim          │
                    │   - Dropout: 0.1            │
                    └─────────────────────────────┘
                                    ↓
                            Global Average Pool
                                    ↓
                            Feed Forward (256)
                                    ↓
                            Output (BBB+/BBB-)
```

### Key Components

- **Custom SMILES Tokenizer**: Character-level tokenization with special tokens
- **Positional Encoding**: Sinusoidal position embeddings
- **Multi-Head Attention**: 8 heads for capturing diverse molecular patterns
- **Layer Normalization**: Stabilizes training
- **Dropout Regularization**: Prevents overfitting

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 256 |
| Hidden Dimension | 512 |
| Number of Layers | 4 |
| Attention Heads | 8 |
| Dropout | 0.1 |
| Max Sequence Length | 128 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |

---

## 📊 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 87.2% |
| **Precision** | 86.5% |
| **Recall** | 88.1% |
| **F1 Score** | 87.3% |
| **AUC-ROC** | 0.92 |


### Endpoints

#### `POST /predict`
Predict BBB penetration for a single molecule.

**Request:**
```json
{
  "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
}
```

**Response:**
```json
{
  "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
  "bbb_score": 0.9234,
  "prediction": "BBB+",
  "confidence": 0.9234,
  "rules_check": null
}
```

#### `POST /predict/batch`
Predict for multiple molecules.

**Request:**
```json
{
  "molecules": [
    {"smiles": "CCO"},
    {"smiles": "c1ccccc1"}
  ]
}
```

**Response:**
```json
{
  "predictions": [...],
  "batch_size": 2,
  "processing_time_ms": 45.2
}
```

#### `GET /health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "SMILESXTransformer (BBB Penetration)",
  "timestamp": "2026-09-29T16:47:31.636Z"
}
```

### Interactive Documentation

Visit http://localhost:8000/docs for the full Swagger UI with interactive testing.

---

## 🧪 Training Your Own Model

### Prepare Data

Place your dataset in `data/raw/` with columns:
- `smiles`: SMILES representation
- `label`: 0 (BBB-) or 1 (BBB+)

### Configure Training

Edit `config/config.yaml`:
```yaml
data:
  train_path: "data/processed/train.csv"
  val_path: "data/processed/val.csv"
  test_path: "data/processed/test.csv"

model:
  embedding_dim: 256
  hidden_dim: 512
  num_layers: 4
  num_heads: 8
  dropout: 0.1
```

### Run Training

```bash
python -m src.training.train --config config/config.yaml
```

## 📈 Future Improvements

- [ ] Add attention visualization
- [ ] Implement molecular fingerprints as additional features
- [ ] Expand to multi-task prediction (BBB + toxicity)
- [ ] Add model explainability (SHAP/LIME)
- [ ] Optimize inference speed with ONNX
- [ ] Add support for 3D molecular structures

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐
