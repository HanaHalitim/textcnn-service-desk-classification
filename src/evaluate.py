import torch
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from . import config
from .data import get_dataloaders
from .model import TextCNN
from .utils import set_seed, get_device

def main():
    set_seed(config.RANDOM_SEED)
    device = get_device()
    
    train_loader, val_loader, test_loader, num_classes, vocab_size = get_dataloaders()
    
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
    
    print(f"Loaded model with best validation accuracy: {checkpoint.get('best_val_accuracy', 'N/A'):.4f}")
    print(f"Evaluating on test set ({len(test_loader.dataset)} samples)...")
    print()
    
    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    prec_metric = Precision(task="multiclass", num_classes=num_classes, average='none').to(device)
    rec_metric = Recall(task="multiclass", num_classes=num_classes, average='none').to(device)
    conf_matrix_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    
    predicted = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            acc_metric.update(preds, labels)
            prec_metric.update(preds, labels)
            rec_metric.update(preds, labels)
            conf_matrix_metric.update(preds, labels)
            
            predicted.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    accuracy = acc_metric.compute().item()
    precision = prec_metric.compute().cpu().tolist()
    recall = rec_metric.compute().cpu().tolist()
    confusion_matrix = conf_matrix_metric.compute().cpu().numpy()
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    metrics_file = os.path.join(config.RESULTS_DIR, "textcnn_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Per-Class Metrics:\n")
        f.write(f"  Precision: {precision}\n")
        f.write(f"  Recall:    {recall}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix))
    
    print("--- Evaluation Metrics ---")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nPer-Class Metrics:")
    print(f"  Precision: {precision}")
    print(f"  Recall:    {recall}")
    print("\nConfusion Matrix (True Labels vs. Predicted Labels):")
    print(confusion_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    conf_matrix_path = os.path.join(config.RESULTS_DIR, "textcnn_confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    
    print(f"\nResults saved to:")
    print(f"  - {metrics_file}")
    print(f"  - {conf_matrix_path}")

if __name__ == "__main__":
    main()

