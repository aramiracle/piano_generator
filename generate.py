import torch

def generate_sequence_from_batch(model, dataloader, gen_seq_len, vocab_size, device, reverse_vocab, one_hot=True):
    """
    Generate a sequence while maintaining the transformation relationship with the source.
    This version handles a wider range of MIDI-compatible events (note_on, note_off, time_shift, etc.)
    """
    model.eval()

    # Extract the first batch from the dataloader
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)

    # Use the first sequence in the batch
    current_src = inputs[0]
    current_tgt = targets[0]
    if one_hot:
        generated = torch.argmax(current_src, dim=-1).tolist()  # Get the most probable class if one-hot
    else:
        generated = current_src.tolist()  # If not one-hot, use token indices directly

    seq_len = len(generated)
    last_note_value = None  # Track the last note value for maintaining relative pitch
    active_notes = {}  # Store active notes by their pitch (for note_off events)
    time_shift = 0  # Track time shifts if needed
    tempo = 120  # Default tempo (beats per minute)

    for _ in range(gen_seq_len):
        # Prepare input sequences for the next chunk
        src = current_src.unsqueeze(0)
        
        if one_hot:
            tgt = current_tgt[:-1, :].unsqueeze(0)
        else:  
            tgt = current_tgt[:-1].unsqueeze(0)

        # Generate the next chunk using generate_sequence
        with torch.no_grad():
            chunk = model.generate_sequence(
                src=src,
                predict_length=1,  # Generate one token at a time, but can be adjusted
                src_mask=None,
                tgt_mask=None,
                memory_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                temperature=1.0
            )

        # Get the next token from the generated chunk
        next_token = chunk[:, -1, :].argmax(dim=-1).item() if one_hot else chunk[:, -1].item()

        # Apply transformation logic to maintain musical relationship
        event_type = reverse_vocab[next_token]

        generated.append(next_token)

        # Convert the predicted token to one-hot encoding (if required)
        if one_hot:
            next_one_hot = torch.zeros(1, vocab_size, device=device)
            next_one_hot[0, next_token] = 1
        else:
            next_note = torch.tensor([next_token], device=device)  # Use token index directly

        # Update sequences for the next chunk generation
        if one_hot:
            current_src = torch.cat([current_src, next_one_hot], dim=0)[1:, :]
        else:
            current_src = torch.cat([current_src, next_note], dim=0)[1:]

    # Return the generated sequence
    return generated[seq_len:]
