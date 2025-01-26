import torch

def generate_sequence_from_batch(model, dataloader, gen_seq_len, vocab_size, device, reverse_vocab, one_hot=True):
    """
    Generate a sequence while maintaining the transformation relationship between source and target.
    Ensures `src` and `tgt` are shifted by one token during generation.

    Args:
        model: Trained Transformer model.
        dataloader: DataLoader providing batches of input data.
        gen_seq_len: Number of additional tokens to generate.
        vocab_size: Size of the vocabulary.
        device: Device for computation (e.g., "cuda" or "cpu").
        reverse_vocab: Dictionary to decode token indices to original representation.
        one_hot: Boolean indicating if the model expects one-hot encoding.

    Returns:
        List of generated token indices.
    """
    model.eval()

    # Extract the first batch from the dataloader
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)

    # Use the first sequence in the batch
    current_src = inputs[0]
    current_tgt = targets[0]

    if one_hot:
        generated = torch.argmax(current_src, dim=-1).tolist()  # Convert one-hot to token indices
    else:
        generated = current_src.tolist()  # Token indices directly

    seq_len = len(generated)

    with torch.no_grad():
        for _ in range(gen_seq_len):
            # Prepare `src` and `tgt` for the model
            src = current_src.unsqueeze(0)  # Add batch dimension
            if one_hot:
                tgt = current_src[1:, :].unsqueeze(0)  # `tgt` is shifted by one token
            else:
                tgt = current_src[1:].unsqueeze(0)

            # Predict the next token
            output = model(
                src=src,
                tgt=tgt,  # Use the shifted sequence as the target
                src_mask=None,
                tgt_mask=None,
                memory_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None
            )

            # Get the next token from the output
            next_token_logits = output[:, -1, :]  # Shape: (batch_size, vocab_size)
            next_token = torch.argmax(next_token_logits, dim=-1).item()  # Get the predicted token index

            # Append the predicted token to the generated sequence
            generated.append(next_token)

            # Convert the predicted token to one-hot (if required)
            if one_hot:
                next_one_hot = torch.zeros(1, vocab_size, device=device)
                next_one_hot[0, next_token] = 1
                current_src = torch.cat([current_src[1:], next_one_hot], dim=0)  # Shift and append
            else:
                next_note = torch.tensor([next_token], device=device)
                current_src = torch.cat([current_src[1:], next_note], dim=0)  # Shift and append

    # Return only the generated sequence
    return generated[seq_len:]

def generate_sequence_from_batch_multinomial(model, dataloader, gen_seq_len, vocab_size, device, reverse_vocab, one_hot=True):
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
