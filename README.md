Binary classification of movie reviews from the IMDB dataset into **positive** and **negative** classes using Python and machine learning.

## Project Overview

- **Dataset**: IMDB reviews (text + label)
- **Task**: binary sentiment classification (positive / negative)
- **Stack**: Python, NumPy, pandas, scikit-learn (and optionally PyTorch / TensorFlow in future)
- **Goal**:
    - build strong baselines using classical ML
    - improve performance using Transformer-based models
    - compare models using consistent metrics
    - create a clean, reproducible ML pipeline

## Tech Stack

### Core
- Python 3.10+
- NumPy
- pandas
- scikit-learn

### NLP & Deep Learning
- PyTorch
- Hugging Face Transformers
- sentence-transformers (MiniLM)
- NLTK

### Experiment Tracking & Visualization
- TensorBoard (NN)
- (optionally) MLflow (ML)

## Installation

```bash
git clone https://github.com/da4nik08/sentiment-binary-classifier.git
cd sentiment-binary-classifier
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preprocessing

Text preprocessing steps include:
- converting text to lowercase
- HTML tag removal
- emoji and unicode symbol removal
- punctuation normalization
- stopword removal (NLTK)
- review length trimming with sentence boundary preservation

## Models

### 1. Classical ML Baselines (Pretrained Embeddings)

For baseline models, **text is encoded using pretrained sentence embeddings**
from the model:

**`sentence-transformers/all-MiniLM-L6-v2`**

The embeddings are generated using the **original pretrained model**
(without fine-tuning), and then used as fixed feature vectors for
classical machine learning classifiers.

**Pipeline**:
1. Text preprocessing
2. Sentence embedding extraction (MiniLM, frozen)
3. Classical ML classifier training

**Used classifiers**:
- Logistic Regression
- Linear SVM
- XGBoost Classifier

This setup allows a fair comparison between:
- classical ML on high-quality semantic embeddings
- end-to-end fine-tuned Transformer model

---

### 2. Transformer-Based Model (Fine-Tuned)

**Model**: `all-MiniLM-L6-v2`

**Architecture**:
- Pretrained MiniLM encoder
- Mean pooling over token embeddings
- Dropout layer
- Fully connected layer (1 neuron)
- BCEWithLogitsLoss

**Training strategy**:
- Encoder weights are mostly frozen
- Last 2 transformer layers are fine-tuned
- Separate learning rates:
  - encoder layers
  - classification head
- Weight decay
- Gradient clipping
- Cosine Annealing learning rate scheduler

---

## Results

### Model Performance Comparison

| Model | Feature Representation | Accuracy | Precision | Recall | F1-score |
|------|------------------------|----------|-----------|--------|----------|
| Logistic Regression | MiniLM embeddings | 0.82986 | 0.82595 | 0.83594 | 0.83089 |
| Linear SVM | MiniLM embeddings | 0.83352 | 0.82757 | 0.84266 | 0.83503 |
| XGBoost | MiniLM embeddings | 0.81952 | 0.81215 | 0.83142 | 0.82166 |
| MiniLM (fine-tuned) | End-to-end Transformer | **0.8972** | **0.8937** | **0.9016** | **0.8976** |

## Tensorboard
**Training vs. Validation Loss**
![Training vs. Validation Loss](https://github.com/da4nik08/sentiment-binary-classifier/blob/feature/analysis-and-docs/assets/tv_loss.png)

**Validation Accuracy**
![Validation Accuracy](https://github.com/da4nik08/sentiment-binary-classifier/blob/feature/analysis-and-docs/assets/val_acc.png)

![Validation F1 score](https://github.com/da4nik08/sentiment-binary-classifier/blob/feature/analysis-and-docs/assets/val_f1.png)

---

### Inference

**Input**:
- Run sentiment inference on a CSV dataset using a fine-tuned MiniLM model.

**Output**
- CSV file with two additional columns:
  - prediction — predicted label (0 or 1)
  - probability — predicted probability of positive sentiment

To run inference on a CSV file:
```bash
python inference.py \
    --csv_path dataset/IMDB_small_test.csv \
    --text_column review_final \
    --config_path configs/inference_config.yaml \
    --output_path predictions.csv

```