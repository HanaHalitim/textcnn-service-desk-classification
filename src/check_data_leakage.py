import numpy as np
from sklearn.model_selection import train_test_split
from .data import load_data, pad_input
from . import config

def check_data_leakage():
    words, text, labels = load_data()
    
    print("Data Leakage Check")
    print("=" * 60)
    print(f"Total samples in dataset: {len(text)}")
    
    unique_texts = set()
    duplicate_count = 0
    for i, tokens in enumerate(text):
        text_str = " ".join(tokens)
        if text_str in unique_texts:
            duplicate_count += 1
        else:
            unique_texts.add(text_str)
    
    print(f"Unique text samples: {len(unique_texts)}")
    print(f"Duplicate text samples: {duplicate_count}")
    
    unique_pairs = set()
    duplicate_pairs = 0
    for i, (tokens, label) in enumerate(zip(text, labels)):
        text_str = " ".join(tokens)
        pair = (text_str, int(label))
        if pair in unique_pairs:
            duplicate_pairs += 1
        else:
            unique_pairs.add(pair)
    
    print(f"Unique (text, label) pairs: {len(unique_pairs)}")
    print(f"Duplicate (text, label) pairs: {duplicate_pairs}")
    print()
    
    word2idx = {o: i for i, o in enumerate(words)}
    
    for i, sentence in enumerate(text):
        text[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]
    
    text_padded = pad_input(text, config.SEQUENCE_LENGTH)
    labels_array = np.array(labels, dtype=np.int64)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        text_padded, labels_array,
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
    
    train_rows = {tuple(row) for row in X_train}
    val_overlap = sum(tuple(row) in train_rows for row in X_val)
    test_overlap = sum(tuple(row) in train_rows for row in X_test)
    
    print("Split Statistics:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print()
    print("Cross-Split Overlap (after padding and indexing):")
    print(f"  Val samples identical to train: {val_overlap} ({100*val_overlap/len(X_val):.2f}%)")
    print(f"  Test samples identical to train: {test_overlap} ({100*test_overlap/len(X_test):.2f}%)")
    print()
    
    if val_overlap > len(X_val) * 0.05 or test_overlap > len(X_test) * 0.05:
        print("WARNING: Found significant overlap across splits!")
        print("   This indicates potential data leakage.")
        if duplicate_pairs > 0:
            print(f"   Root cause: {duplicate_pairs} duplicate (text, label) pairs in original dataset.")
    else:
        print("OK: Minimal overlap across splits (< 5%).")
        if duplicate_pairs == 0:
            print("   All (text, label) pairs are unique in the original dataset.")
    
    print()
    print("Class distribution:")
    print(f"  Train: {np.bincount(y_train)}")
    print(f"  Val:   {np.bincount(y_val)}")
    print(f"  Test:  {np.bincount(y_test)}")

if __name__ == "__main__":
    check_data_leakage()

