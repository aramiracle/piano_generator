import torch
import random
from utils import generate_square_subsequent_mask

def generate_sequence(model, start_seq_size, max_length, vocab_size, device):
    model.eval()

    # Generate a random starting sequence as one-hot encoded tokens
    start_seq = [random.randint(0, vocab_size - 1) for _ in range(start_seq_size)]
    
    # Convert to one-hot encoding (shape: (seq_length, vocab_size))
    one_hot_start_seq = torch.zeros(start_seq_size, vocab_size, device=device)
    for idx, token in enumerate(start_seq):
        one_hot_start_seq[idx, token] = 1

    generated = start_seq
    for _ in range(max_length):
        # Prepare src (input sequence) and tgt (target sequence shifted by one)
        src = one_hot_start_seq.unsqueeze(0)  # Add batch dimension (shape: (1, seq_length, vocab_size))
        src_mask = generate_square_subsequent_mask(src.size(1), device).to(device)  # Create mask
        
        # tgt is the same as src but with the generated sequence
        tgt = torch.zeros_like(src, device=device)
        tgt[:, :-1, :] = src[:, 1:, :]  # Shift by one for the target sequence

        tgt_mask = generate_square_subsequent_mask(tgt.size(1), device).to(device)  # Create mask
        
        with torch.no_grad():
            output = model(src, tgt, src_mask, tgt_mask)  # Forward pass with different src and tgt
        
        # Get the most probable next token from the model's output (shape: [1, seq_length, vocab_size])
        next_token = torch.argmax(output[:, -1, :], dim=-1).item()  # Get the index of the most probable token
        
        # Append the predicted token index to the sequence
        generated.append(next_token)

        # Convert the predicted token to a one-hot vector and append it to the sequence for the next iteration
        next_one_hot = torch.zeros(1, vocab_size, device=device)
        next_one_hot[0, next_token] = 1
        one_hot_start_seq = torch.cat([one_hot_start_seq, next_one_hot], dim=0)  # Update input sequence

        # Check for end-of-sequence token
        if next_token == vocab_size - 1:  # End-of-sequence token
            break
    
    return generated
