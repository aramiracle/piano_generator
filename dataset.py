import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

class MusicDataset(Dataset):
    def __init__(self, sequences, vocab_size, one_hot=True):
        self.sequences = sequences
        self.vocab_size = vocab_size
        self.one_hot = one_hot
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Input sequence is all tokens except the last one
        src = sequence[:-1]
        # Target sequence is all tokens except the first one
        tgt = sequence[1:]

        if self.one_hot:
            # Convert to one-hot encoding
            src = self.one_hot_encode(src)
            tgt = self.one_hot_encode(tgt)

        return torch.tensor(src, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32) if self.one_hot else torch.tensor(src, dtype=torch.int64), torch.tensor(tgt, dtype=torch.int64)
    
    def one_hot_encode(self, sequence):
        one_hot = np.zeros((len(sequence), self.vocab_size), dtype=np.float32)
        for i, index in enumerate(sequence):
            one_hot[i, index] = 1.0
        return one_hot

def get_dataloader(preprocessed_path, batch_size=16, one_hot=True):
    # Load preprocessed data
    with open(preprocessed_path, 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['src_sequences']  # Use source sequences for training
    vocab = data['vocab']
    reverse_vocab = data['reverse_vocab']
    vocab_size = len(vocab)

    # Create dataset
    dataset = MusicDataset(sequences, vocab_size, one_hot)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, vocab, reverse_vocab
