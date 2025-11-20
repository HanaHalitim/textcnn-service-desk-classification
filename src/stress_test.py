import json
import numpy as np
import torch
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix as sklearn_confusion_matrix
from . import config
from .data import load_data, pad_input, reconstruct_text_from_tokens
from .model import TextCNN
from .utils import set_seed, get_device
from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix

nltk.download('punkt_tab', quiet=True)

def preprocess_text(text, word2idx, seq_len):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum()]
    indices = [word2idx.get(token, 0) for token in tokens]
    padded = pad_input([indices], seq_len)
    return torch.LongTensor(padded)

def train_baseline_model():
    print("Training baseline model for stress test comparison...")
    words, tokenized_texts, labels = load_data()
    texts = reconstruct_text_from_tokens(tokenized_texts)
    labels_array = np.array(labels, dtype=np.int64)
    
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
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_SEED,
        multi_class='multinomial',
        solver='lbfgs',
        C=1.0
    )
    
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

def main():
    set_seed(config.RANDOM_SEED)
    device = get_device()
    
    stress_test_path = os.path.join(config.DATA_DIR, "stress_test.json")
    if not os.path.exists(stress_test_path):
        print(f"Error: Stress test file not found at {stress_test_path}")
        return
    
    print("Loading stress test data...")
    with open(stress_test_path, 'r') as f:
        stress_data = json.load(f)
    
    print(f"Loaded {len(stress_data)} stress test samples")
    
    print("\nLoading TextCNN model...")
    words, _, _ = load_data()
    word2idx = {o: i for i, o in enumerate(words)}
    vocab_size = len(word2idx) + 1
    num_classes = 5
    
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=config.EMBEDDING_DIM,
        num_classes=num_classes,
        filter_count=config.FILTER_COUNT,
        kernel_sizes=config.KERNEL_SIZES,
        dropout_rate=config.DROPOUT_RATE
    )
    
    model_path = os.path.join(config.SAVED_MODELS_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run training first: python -m src.train")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (best val accuracy: {checkpoint.get('best_val_accuracy', 'N/A'):.4f})")
    
    class_names = [
        "Access/Login Issues",
        "Software Bugs/Errors",
        "Hardware/Device Failure",
        "Network/System Outage",
        "New Request/Feature"
    ]
    
    predictions = []
    true_labels = []
    texts = []
    
    print("\nRunning predictions...")
    with torch.no_grad():
        for item in stress_data:
            text = item['text']
            true_label = int(item['label'])
            
            input_tensor = preprocess_text(text, word2idx, config.SEQUENCE_LENGTH)
            input_tensor = input_tensor.to(device)
            
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_label = pred.item()
            
            texts.append(text)
            true_labels.append(true_label)
            predictions.append(pred_label)
    
    accuracy = sum(1 for t, p in zip(true_labels, predictions) if t == p) / len(true_labels)
    
    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    prec_metric = Precision(task="multiclass", num_classes=num_classes, average='none').to(device)
    rec_metric = Recall(task="multiclass", num_classes=num_classes, average='none').to(device)
    conf_matrix_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    
    true_tensor = torch.tensor(true_labels, device=device)
    pred_tensor = torch.tensor(predictions, device=device)
    
    acc_metric.update(pred_tensor, true_tensor)
    prec_metric.update(pred_tensor, true_tensor)
    rec_metric.update(pred_tensor, true_tensor)
    conf_matrix_metric.update(pred_tensor, true_tensor)
    
    precision = prec_metric.compute().cpu().tolist()
    recall = rec_metric.compute().cpu().tolist()
    confusion_matrix = conf_matrix_metric.compute().cpu().numpy()
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    predictions_file = os.path.join(config.RESULTS_DIR, "stress_test_predictions.txt")
    with open(predictions_file, 'w') as f:
        f.write("Stress Test Results - TextCNN Predictions\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({int(accuracy * len(true_labels))}/{len(true_labels)} correct)\n\n")
        f.write("Per-Example Predictions:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Text':<50} {'True':<25} {'Pred':<25} {'Match'}\n")
        f.write("-" * 80 + "\n")
        
        for i, (text, true_l, pred_l) in enumerate(zip(texts, true_labels, predictions)):
            match = "CORRECT" if true_l == pred_l else "WRONG"
            true_name = f"{true_l} ({class_names[true_l]})"
            pred_name = f"{pred_l} ({class_names[pred_l]})"
            text_short = text[:47] + "..." if len(text) > 50 else text
            f.write(f"{text_short:<50} {true_name:<25} {pred_name:<25} {match}\n")
    
    metrics_file = os.path.join(config.RESULTS_DIR, "stress_test_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Stress Test Evaluation - TextCNN\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples: {len(stress_data)}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Per-Class Metrics:\n")
        f.write(f"  Precision: {precision}\n")
        f.write(f"  Recall:    {recall}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix) + "\n\n")
        f.write("-" * 60 + "\n")
        f.write("Summary:\n")
        f.write("-" * 60 + "\n")
        f.write(f"The TextCNN model achieved {accuracy*100:.2f}% accuracy on the stress test set.\n")
        f.write("This stress test contains realistic, messy tickets with typos, mixed phrasing,\n")
        f.write("and overlapping categories to probe generalization beyond the synthetic training data.\n")
    
    print("\n" + "="*60)
    print("Stress Test Results")
    print("="*60)
    print(f"Overall Accuracy: {accuracy:.4f} ({int(accuracy * len(true_labels))}/{len(true_labels)} correct)")
    print("\nPer-Class Metrics:")
    print(f"  Precision: {precision}")
    print(f"  Recall:    {recall}")
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    
    print("\n" + "-"*60)
    print("Sample Predictions (first 10):")
    print("-"*60)
    for i in range(min(10, len(texts))):
        match = "[CORRECT]" if true_labels[i] == predictions[i] else "[WRONG]"
        print(f"{match} Text: {texts[i][:60]}...")
        print(f"  True: {true_labels[i]} ({class_names[true_labels[i]]})")
        print(f"  Pred: {predictions[i]} ({class_names[predictions[i]]})")
        print()
    
    print("\n" + "="*60)
    print("Evaluating Baseline Model on Stress Test")
    print("="*60)
    
    baseline_model, baseline_vectorizer = train_baseline_model()
    
    stress_texts = [item['text'] for item in stress_data]
    stress_labels_baseline = [int(item['label']) for item in stress_data]
    
    stress_tfidf = baseline_vectorizer.transform(stress_texts)
    baseline_predictions = baseline_model.predict(stress_tfidf)
    baseline_accuracy = accuracy_score(stress_labels_baseline, baseline_predictions)
    baseline_precision, baseline_recall, _, _ = precision_recall_fscore_support(
        stress_labels_baseline, baseline_predictions, average=None, zero_division=0
    )
    baseline_confusion = sklearn_confusion_matrix(stress_labels_baseline, baseline_predictions)
    
    print(f"Baseline Test Accuracy: {baseline_accuracy:.4f} ({int(baseline_accuracy * len(stress_data))}/{len(stress_data)} correct)")
    print(f"TextCNN Test Accuracy: {accuracy:.4f} ({int(accuracy * len(true_labels))}/{len(true_labels)} correct)")
    print(f"Difference: {(accuracy - baseline_accuracy)*100:+.2f} percentage points")
    
    with open(metrics_file, 'a') as f:
        f.write("\n" + "="*60 + "\n")
        f.write("Baseline Model (TF-IDF + Logistic Regression) on Stress Test\n")
        f.write("="*60 + "\n\n")
        f.write(f"Baseline Accuracy: {baseline_accuracy:.4f}\n")
        f.write(f"TextCNN Accuracy: {accuracy:.4f}\n")
        f.write(f"Difference: {(accuracy - baseline_accuracy)*100:+.2f} percentage points\n\n")
        f.write("Baseline Per-Class Metrics:\n")
        f.write(f"  Precision: {baseline_precision.tolist()}\n")
        f.write(f"  Recall:    {baseline_recall.tolist()}\n\n")
        f.write("Baseline Confusion Matrix:\n")
        f.write(str(baseline_confusion) + "\n\n")
        f.write("-"*60 + "\n")
        f.write("Comparison Summary:\n")
        f.write("-"*60 + "\n")
        if accuracy > baseline_accuracy:
            f.write(f"TextCNN outperforms the baseline by {(accuracy - baseline_accuracy)*100:.2f} percentage points ")
            f.write("on the stress test set, suggesting better robustness to typos and phrasing variation.\n")
        elif baseline_accuracy > accuracy:
            f.write(f"Baseline outperforms TextCNN by {(baseline_accuracy - accuracy)*100:.2f} percentage points on the stress test set.\n")
        else:
            f.write("Both models perform equally on the stress test set.\n")
    
    print(f"\nResults saved to:")
    print(f"  - {predictions_file}")
    print(f"  - {metrics_file}")

if __name__ == "__main__":
    main()

