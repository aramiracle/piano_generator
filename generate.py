import torch
from utils import generate_square_subsequent_mask

def generate_sequence(model, start_seq, max_length, vocab_size, device):
    model.eval()
    generated = start_seq
    for _ in range(max_length):
        src = torch.tensor(generated).unsqueeze(0).to(device)
        src_mask = generate_square_subsequent_mask(src.size(1)).to(device)
        with torch.no_grad():
            output = model(src, src, src_mask, src_mask)
        next_token = torch.argmax(output[:, -1, :], dim=-1).item()
        generated.append(next_token)
        if next_token == vocab_size - 1:  # End-of-sequence token
            break
    return generated
