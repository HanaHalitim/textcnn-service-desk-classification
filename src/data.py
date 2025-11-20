import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from . import config

def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def reconstruct_text_from_tokens(tokenized_texts):
    texts = []
    for tokens in tokenized_texts:
        text = " ".join(tokens)
        texts.append(text)
    return texts

def load_data(data_dir=None):
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    words_path = os.path.join(data_dir, "words.json")
    text_path = os.path.join(data_dir, "text.json")
    labels_path = os.path.join(data_dir, "labels.npy")
    
    with open(words_path, 'r') as f:
        words = json.load(f)
    with open(text_path, 'r') as f:
        text = json.load(f)
    labels = np.load(labels_path)
    
    return words, text, labels

def get_dataloaders(seq_len=None, batch_size=None, test_size=0.15, val_size=0.15, random_state=None):
    if seq_len is None:
        seq_len = config.SEQUENCE_LENGTH
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if random_state is None:
        random_state = config.RANDOM_SEED
    
    words, text, labels = load_data()
    
    word2idx = {o: i for i, o in enumerate(words)}
    
    for i, sentence in enumerate(text):
        text[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]
    
    text_padded = pad_input(text, seq_len)
    labels_array = np.array(labels, dtype=np.int64)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        text_padded, labels_array, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels_array
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    train_dataset = TensorDataset(
        torch.LongTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.LongTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.LongTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    num_classes = len(np.unique(labels))
    vocab_size = len(word2idx) + 1
    
    return train_loader, val_loader, test_loader, num_classes, vocab_size

