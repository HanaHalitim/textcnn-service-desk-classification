import torch
import torch.nn as nn
from torchmetrics import Accuracy
import os
import json
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
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1
    )
    
    val_acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    log_file = os.path.join(config.RESULTS_DIR, "textcnn_training_log.txt")
    history_file = os.path.join(config.RESULTS_DIR, "textcnn_training_history.json")
    
    with open(log_file, 'w') as f:
        f.write("Training Log\n")
        f.write("=" * 50 + "\n\n")
    
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting Training for up to {config.EPOCHS} epochs...")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE} epochs")
    print()
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        val_acc_metric.reset()
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                val_loss += batch_loss.item()
                _, preds = torch.max(outputs, 1)
                val_acc_metric.update(preds, labels)
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_acc_metric.compute().item()
        val_acc_metric.reset()
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(avg_val_loss)
        
        log_msg = f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
        
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            patience_counter = 0
            log_msg += " <--- New Best (val loss improved)!"
            
            model_path = os.path.join(config.SAVED_MODELS_DIR, "best_model.pt")
            torch.save({
                'model_state_dict': best_model_state,
                'config': {
                    'vocab_size': vocab_size,
                    'embed_dim': config.EMBEDDING_DIM,
                    'num_classes': num_classes,
                    'filter_count': config.FILTER_COUNT,
                    'kernel_sizes': config.KERNEL_SIZES,
                    'dropout_rate': config.DROPOUT_RATE,
                },
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
                'epoch': epoch + 1
            }, model_path)
        else:
            patience_counter += 1
            log_msg += f" | Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                log_msg += " [EARLY STOPPING]"
        
        print(log_msg)
        
        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break
    
    print("\nTraining finished.")
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"Model saved to: {os.path.join(config.SAVED_MODELS_DIR, 'best_model.pt')}")
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_accuracy,
        'epochs_trained': len(train_losses)
    }
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved to: {history_file}")

if __name__ == "__main__":
    main()

