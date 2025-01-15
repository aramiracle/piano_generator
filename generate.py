import torch

def generate_sequence_from_batch(model, dataloader, gen_seq_len, vocab_size, device, reverse_vocab):
    """
    Generate a sequence while maintaining the transformation relationship with the source.
    """
    model.eval()

    # Extract the first batch from the dataloader
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)

    # Use the first sequence in the batch
    current_src = inputs[0]
    current_tgt = targets[0]
    generated = torch.argmax(current_src, dim=-1).tolist()
    
    seq_len = len(generated)
    last_note_value = None  # Track the last note value for maintaining relative pitch

    for _ in range(gen_seq_len):
        # Prepare input sequences
        src = current_src.unsqueeze(0)
        tgt = current_tgt[:-1, :].unsqueeze(0)

        with torch.no_grad():
            output = model(src, tgt)

        # Get the most probable next token
        next_token = torch.argmax(output[:, -1, :], dim=-1).item()
        
        # Apply transformation logic to maintain musical relationship
        event_type = reverse_vocab[next_token]
        if event_type.startswith("note_"):
            current_pitch = int(event_type.split("_")[1])
            if last_note_value is None:
                # First note - use as reference
                last_note_value = current_pitch
            else:
                # Maintain relative pitch movement
                pitch_diff = current_pitch - last_note_value
                current_pitch = (last_note_value + pitch_diff) % 88  # Wrap around within range
                last_note_value = current_pitch
                
                # Update next_token based on transformed pitch
                next_token = [k for k, v in reverse_vocab.items() if v == f"note_{current_pitch}"][0]
        elif event_type.startswith("time_"):
            # Keep timing events as is
            last_note_value = None  # Reset pitch tracking after time events

        generated.append(next_token)

        # Convert the predicted token to one-hot encoding
        next_one_hot = torch.zeros(1, vocab_size, device=device)
        next_one_hot[0, next_token] = 1

        # Update sequences
        current_src = torch.cat([current_src, next_one_hot], dim=0)[-seq_len:, :]
        current_tgt = torch.cat([current_tgt, next_one_hot], dim=0)[-seq_len:, :]

    return generated[seq_len:]