# Service Desk Ticket Classification with TextCNN (PyTorch)

A PyTorch implementation of a TextCNN (Convolutional Neural Network for Text Classification) model for classifying IT service desk tickets into 5 categories.

## Project Overview

This project implements a deep learning solution for automatically categorizing IT service desk tickets. The model uses a Convolutional Neural Network architecture specifically designed for text classification, combining word embeddings with multi-kernel convolutions and max pooling to capture different n-gram patterns in ticket descriptions.

## Task Description

The goal is to classify short IT service desk ticket texts into one of 5 categories:
- **Class 0**: Access/Login Issues
- **Class 1**: Software Bugs/Errors
- **Class 2**: Hardware/Device Failure
- **Class 3**: Network/System Outage
- **Class 4**: New Request/Feature

## Dataset

The project uses a synthetic dataset of IT service desk tickets. The dataset consists of:
- **4,000 ticket samples** (800 per class)
- 5 balanced classes
- Preprocessed and tokenized text data
- **Vocabulary size: ~578 unique words**

The dataset is generated synthetically using templates and variations to create realistic ticket descriptions. The generation process ensures **uniqueness of (text, label) pairs** to prevent data leakage (see Data Leakage Story section below).

## Models

### TextCNN (PyTorch)

The main deep learning model uses a Convolutional Neural Network architecture specifically designed for text classification:

1. **Embedding Layer**: Maps word indices to dense vector representations (embedding_dim=50)
2. **Convolutional Layers**: Multiple parallel 1D convolutions with different kernel sizes (3, 4, 5) to capture n-gram patterns
3. **Max Pooling**: Global max pooling over the sequence length for each convolution
4. **Concatenation**: Combines features from all kernel sizes
5. **Dropout**: Regularization layer (dropout_rate=0.5)
6. **Fully Connected Layer**: Maps concatenated features to class scores

**Hyperparameters:**
- Sequence Length: 50 tokens
- Embedding Dimension: 50
- Filter Count: 64 per kernel
- Kernel Sizes: [3, 4, 5]
- Batch Size: 32
- Learning Rate: 0.001
- Epochs: 20 (with early stopping)
- Dropout Rate: 0.5
- Weight Decay: 1e-4 (L2 regularization)

### Baseline (TF-IDF + Logistic Regression)

A classical machine learning baseline using:
- **TF-IDF Vectorization**: Term frequency-inverse document frequency with n-grams (1-2)
- **Logistic Regression**: Multinomial logistic regression classifier

This baseline serves as a reference point to compare against the deep learning approach and demonstrates the benefit of learned representations.

## Training Procedure

The TextCNN model is trained using:
- **Optimizer**: Adam with weight decay (L2 regularization)
- **Loss Function**: CrossEntropyLoss
- **Early Stopping**: Model checkpointing based on **validation loss** (not just accuracy) with patience of 3 epochs
- **Learning Rate Scheduling**: ReduceLROnPlateau scheduler
- **Train/Val/Test Split**: 70%/15%/15% (stratified, with **0% overlap** between splits)

Training logs are saved to `results/training_log.txt` and the best model is saved to `saved_models/best_model.pt`.

## Baseline vs TextCNN Comparison

The baseline model (TF-IDF + Logistic Regression) serves as a classical ML reference point for fair comparison:

### Main Test Set Performance

- **Baseline Test Accuracy**: 98.00%
- **TextCNN Test Accuracy**: 98.00%
- **Difference**: 0.00% (tied)

**On this synthetic, well-structured dataset, both models perform equally well.** This is actually a **positive outcome** that demonstrates:

1. **Fair comparison**: We didn't force a deep model to "win" - we built a strong baseline and compared honestly
2. **Task characteristics**: The decision boundary in bag-of-words space is nearly linearly separable, so TF-IDF features are sufficient
3. **Methodology**: Having both deep and classical models allows us to understand when each approach is appropriate

The fact that both models tie at 98% suggests that for this particular synthetic dataset, the class boundaries are driven by clear keywords and patterns that both approaches can capture effectively.

## Stress Test Set

To probe generalization beyond the synthetic training distribution, we created a **stress test set** of 50 manually written, realistic tickets with:

- **Typos and misspellings**: `"cant logn into my mail since yestreday"`
- **Mixed phrasing**: `"Hi, after windows update my vpn stopped working and I can't access intranet"`
- **Extra chatter**: `"hey team, sorry to bother again but my printer is dead again... power light blinking etc."`
- **Overlapping categories**: Tickets that could reasonably belong to multiple classes

The stress test evaluates how well the model handles:
- Real-world noise and imperfections
- Natural language variations
- Ambiguous cases

**Important Note**: With only 50 samples, this is a **qualitative robustness check** rather than a strong statistical benchmark. Small changes in predictions can significantly affect the percentage.

### Stress Test Results

- **TextCNN Accuracy**: 96.00% (48/50 correct)
- **Baseline Accuracy**: 98.00% (49/50 correct)
- **Difference**: Baseline outperforms by 2.00 percentage points

**Error Analysis - TextCNN:**
- **Success example**: Correctly classified `"cant logn into my mail since yestreday"` (typos) as Access/Login Issues
- **Failure example**: Misclassified `"Hi, after windows update my vpn stopped working and I can't access intranet"` as Access/Login (Class 0) instead of Network/System Outage (Class 3)
  - This is an ambiguous case where "can't access" suggests login issues, but "vpn stopped working" indicates network problems

**Common error patterns:**
- Confusion between Access/Login Issues (Class 0) and Network/System Outage (Class 3) - both involve "access" and "connection" concepts
- Both models handle typos and natural language variations reasonably well

This demonstrates that while both models generalize well to noisy text, there are inherent challenges with ambiguous real-world tickets that span multiple categories. The stress test serves as a **qualitative validation** that the models can handle realistic imperfections, not a definitive performance benchmark.

## Repository Structure

```
Service_Desk/
│
├── data/
│   ├── words.json          # Vocabulary list
│   ├── text.json           # Tokenized ticket texts
│   ├── labels.npy          # Class labels
│   ├── stress_test.json    # Stress test set (messy realistic tickets)
│   └── README.md           # Data documentation
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Hyperparameters and configuration
│   ├── data.py             # Data loading and preprocessing
│   ├── model.py            # TextCNN model definition
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── baseline.py         # Baseline model (TF-IDF + Logistic Regression)
│   ├── stress_test.py      # Stress test evaluation script
│   ├── check_data_leakage.py  # Data leakage verification
│   ├── plot_training_curves.py # Training visualization
│   └── utils.py            # Utility functions
│
├── saved_models/           # Saved model checkpoints
│   └── best_model.pt
│
├── results/                # Training logs and evaluation results
│   ├── textcnn_training_log.txt
│   ├── textcnn_training_history.json
│   ├── textcnn_training_curves.png
│   ├── textcnn_metrics.txt
│   ├── textcnn_confusion_matrix.png
│   ├── baseline_metrics.txt
│   ├── baseline_confusion_matrix.png
│   ├── stress_test_predictions.txt
│   └── stress_test_metrics.txt
│
├── generate_data.py        # Script to generate synthetic dataset
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore rules
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Service_Desk
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate the dataset (if not already present):
```bash
python generate_data.py
```

## Usage

### 1. Generate Dataset

Generate the synthetic ticket dataset:

```bash
python generate_data.py
```

This creates `data/words.json`, `data/text.json`, and `data/labels.npy`.

### 2. Train the Model

Train the TextCNN model:

```bash
python -m src.train
```

Or:

```bash
python src/train.py
```

The training script will:
- Load and preprocess the data
- Create train/val/test splits
- Train the model for the specified number of epochs
- Save the best model based on validation accuracy
- Log training progress to `results/textcnn_training_log.txt`

### 3. Evaluate the Model

Evaluate the trained TextCNN model on the test set:

```bash
python -m src.evaluate
```

Or:

```bash
python src/evaluate.py
```

The evaluation script will:
- Load the best saved model
- Evaluate on the test set
- Compute accuracy, precision, recall, and confusion matrix
- Save metrics to `results/textcnn_metrics.txt`
- Generate and save confusion matrix plot to `results/textcnn_confusion_matrix.png`

### 4. Run Baseline Model

Train and evaluate the baseline (TF-IDF + Logistic Regression) for comparison:

```bash
python -m src.baseline
```

Or:

```bash
python src/baseline.py
```

This will:
- Train a TF-IDF + Logistic Regression model on the same dataset
- Evaluate on validation and test sets
- Save metrics to `results/baseline_metrics.txt`
- Generate confusion matrix plot to `results/baseline_confusion_matrix.png`
- Compare results with TextCNN

### 5. Run Stress Test Evaluation

Evaluate the TextCNN model on the realistic stress test set:

```bash
python -m src.stress_test
```

Or:

```bash
python src/stress_test.py
```

This will:
- Load the best TextCNN model
- Evaluate on the stress test set (50 messy, realistic tickets)
- Show per-example predictions
- Save detailed results to `results/stress_test_predictions.txt`
- Save summary metrics to `results/stress_test_metrics.txt`

### 6. Check Data Leakage (Optional)

Verify that there's no data leakage between train/val/test splits:

```bash
python -m src.check_data_leakage
```

### 7. Plot Training Curves (Optional)

Visualize training progress:

```bash
python -m src.plot_training_curves
```

This generates `results/textcnn_training_curves.png` with train/val loss and accuracy plots.

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- torchmetrics
- numpy
- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn

See `requirements.txt` for specific versions.

## Configuration

All hyperparameters and configuration settings can be modified in `src/config.py`:

```python
SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 50          # Reduced to prevent overfitting
FILTER_COUNT = 64           # Reduced to prevent overfitting
KERNEL_SIZES = [3, 4, 5]
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20                 # With early stopping
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 1e-4         # L2 regularization
EARLY_STOPPING_PATIENCE = 3
```
