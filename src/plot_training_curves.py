import json
import matplotlib.pyplot as plt
import os
from . import config

def plot_training_curves():
    history_file = os.path.join(config.RESULTS_DIR, "textcnn_training_history.json")
    
    if not os.path.exists(history_file):
        print(f"Error: Training history not found at {history_file}")
        print("Please run training first: python -m src.train")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    val_accuracies = history['val_accuracies']
    epochs = range(1, len(train_losses) + 1)
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, train_losses, label="Train Loss", marker='o')
    axes[0].plot(epochs, val_losses, label="Val Loss", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Val Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, val_accuracies, label="Val Accuracy", marker='o', color='green')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    plot_path = os.path.join(config.RESULTS_DIR, "textcnn_training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {plot_path}")
    
    if len(train_losses) > 1:
        loss_ratio = [t/v if v > 0 else 0 for t, v in zip(train_losses, val_losses)]
        print("\nTrain/Val Loss Ratio (should stay < 2-3):")
        for i, ratio in enumerate(loss_ratio, 1):
            status = "WARNING" if ratio > 2.5 else "OK"
            print(f"  Epoch {i}: {ratio:.3f} ({status})")

if __name__ == "__main__":
    plot_training_curves()

