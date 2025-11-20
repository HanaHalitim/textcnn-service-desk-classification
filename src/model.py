import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_count, kernel_sizes, dropout_rate):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                      out_channels=filter_count, 
                      kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        
        total_features = filter_count * len(kernel_sizes) 
        self.fc = nn.Linear(total_features, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1) 
        
        pooled_outputs = []
        for conv in self.convs:
            out = F.relu(conv(x))
            out = F.max_pool1d(out, kernel_size=out.size(2)).squeeze(2) 
            pooled_outputs.append(out)
            
        x = torch.cat(pooled_outputs, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


