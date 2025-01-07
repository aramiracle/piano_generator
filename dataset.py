# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

class MusicDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Input sequence is all tokens except the last one
        src = torch.tensor(sequence[:-1], dtype=torch.long)
        # Target sequence is all tokens except the first one
        tgt = torch.tensor(sequence[1:], dtype=torch.long)

        return src, tgt

def get_dataloader(preprocessed_path, batch_size=16):
    # Load preprocessed data
    with open(preprocessed_path, 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    vocab = data['vocab']
    reverse_vocab = data['reverse_vocab']
    
    # Create dataset
    dataset = MusicDataset(sequences)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, vocab, reverse_vocab