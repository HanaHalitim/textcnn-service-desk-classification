import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .data import load_data
from . import config

def reconstruct_text_from_tokens(tokenized_texts):
    texts = []
    for tokens in tokenized_texts:
        text = " ".join(tokens)
        texts.append(text)
    return texts

def main():
    print("Loading dataset...")
    words, tokenized_texts, labels = load_data()
    
    texts = reconstruct_text_from_tokens(tokenized_texts)
    labels_array = np.array(labels, dtype=np.int64)
    
    print(f"Total samples: {len(texts)}")
    print(f"Classes: {len(np.unique(labels_array))}")
    
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels_array,
        test_size=0.15,
        random_state=config.RANDOM_SEED,
        stratify=labels_array
    )
    
    val_size_adjusted = 0.15 / (1 - 0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=config.RANDOM_SEED,
        stratify=y_temp
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    print("\n" + "="*60)
    print("Training TF-IDF + Logistic Regression Baseline")
    print("="*60)
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        max_df=0.95
    )
    
    print("\nFitting TF-IDF vectorizer...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_SEED,
        multi_class='multinomial',
        solver='lbfgs',
        C=1.0
    )
    
    model.fit(X_train_tfidf, y_train)
    
    print("\nEvaluating on validation set...")
    y_val_pred = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision, val_recall, _, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average=None, zero_division=0
    )
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    print("\nEvaluating on test set...")
    y_test_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision, test_recall, _, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average=None, zero_division=0
    )
    test_confusion = confusion_matrix(y_test, y_test_pred)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nPer-Class Metrics (Test Set):")
    print(f"  Precision: {test_precision}")
    print(f"  Recall:    {test_recall}")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    metrics_file = os.path.join(config.RESULTS_DIR, "baseline_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Baseline Model (TF-IDF + Logistic Regression) - Evaluation Metrics\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write("Per-Class Metrics (Test Set):\n")
        f.write(f"  Precision: {test_precision.tolist()}\n")
        f.write(f"  Recall:    {test_recall.tolist()}\n\n")
        f.write("Confusion Matrix (Test Set):\n")
        f.write(str(test_confusion) + "\n\n")
        f.write("-" * 70 + "\n")
        f.write("Comparison with TextCNN\n")
        f.write("-" * 70 + "\n")
        f.write("(Compare these numbers with results/textcnn_metrics.txt for TextCNN)\n")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(5)],
                yticklabels=[f'Class {i}' for i in range(5)])
    plt.title('Baseline Model - Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    conf_matrix_path = os.path.join(config.RESULTS_DIR, "baseline_confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    
    print(f"\nResults saved to:")
    print(f"  - {metrics_file}")
    print(f"  - {conf_matrix_path}")
    
    try:
        textcnn_metrics_file = os.path.join(config.RESULTS_DIR, "textcnn_metrics.txt")
        if os.path.exists(textcnn_metrics_file):
            with open(textcnn_metrics_file, 'r') as f:
                textcnn_content = f.read()
                if "Overall Accuracy:" in textcnn_content:
                    for line in textcnn_content.split('\n'):
                        if "Overall Accuracy:" in line:
                            textcnn_acc = float(line.split(':')[1].strip())
                            break
                    
                    with open(metrics_file, 'a') as f:
                        f.write(f"\nBaseline Test Accuracy: {test_accuracy:.4f}\n")
                        f.write(f"TextCNN Test Accuracy: {textcnn_acc:.4f}\n")
                        diff = textcnn_acc - test_accuracy
                        f.write(f"Difference: {diff:+.4f} ({diff*100:+.2f}%)\n\n")
                        if diff > 0:
                            f.write(f"TextCNN outperforms the baseline by {diff*100:.2f} percentage points, ")
                            f.write("showing the benefit of learned representations for this task.\n")
                        else:
                            f.write(f"Baseline outperforms TextCNN by {abs(diff)*100:.2f} percentage points.\n")
                    
                    print(f"\nComparison:")
                    print(f"  Baseline Test Accuracy: {test_accuracy:.4f}")
                    print(f"  TextCNN Test Accuracy: {textcnn_acc:.4f}")
                    print(f"  Difference: {diff:+.4f} ({diff*100:+.2f}%)")
    except Exception as e:
        print(f"\nNote: Could not load TextCNN metrics for comparison: {e}")

if __name__ == "__main__":
    main()

