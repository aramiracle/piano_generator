import torch
import pickle

def generate_square_subsequent_mask(size, device):
    mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
    return ~mask

def load_vocab_size(preprocessed_path):
    with open(preprocessed_path, 'rb') as f:
        data = pickle.load(f)
    return len(data['vocab'])