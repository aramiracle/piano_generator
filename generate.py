import torch

def generate_sequence_from_batch(model, dataloader, gen_seq_len, vocab_size, device):
    model.eval()

    # Extract the first batch from the dataloader
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)

    # Use the first sequence in the batch
    current_src = inputs[0]  # Shape: (seq_len, vocab_size)
    current_tgt = targets[0]
    generated = torch.argmax(current_src, dim=-1).tolist()  # Convert initial sequence to indices

    seq_len = len(generated)

    for _ in range(gen_seq_len):  # Generate additional notes
        # Prepare src (input sequence)
        src = current_src.unsqueeze(0)  # Add batch dimension (shape: (1, seq_len, vocab_size))
        tgt = current_tgt[:-1, :].unsqueeze(0)

        with torch.no_grad():
            output = model(src, tgt)  # Predict the next token

        # Get the most probable next token
        next_token = torch.argmax(output[:, -1, :], dim=-1).item()
        generated.append(next_token)

        # Convert the predicted token to one-hot encoding
        next_one_hot = torch.zeros(1, vocab_size, device=device)
        next_one_hot[0, next_token] = 1

        # Concatenate the new token and slide the sequence window
        current_src = torch.cat([current_src, next_one_hot], dim=0)[-seq_len:, :]
        current_tgt = torch.cat([current_tgt, next_one_hot], dim=0)[-seq_len:, :]

    return generated[seq_len:]
