import torch

def generate_sequence_from_batch(model, dataloader, gen_seq_len, vocab_size, device, reverse_vocab):
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
    generated = torch.argmax(current_src, dim=-1).tolist()

    seq_len = len(generated)
    last_note_value = None  # Track the last note value for maintaining relative pitch
    active_notes = {}  # Store active notes by their pitch (for note_off events)
    time_shift = 0  # Track time shifts if needed
    tempo = 120  # Default tempo (beats per minute)

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
            if event_type.startswith("note_on_"):
                # Handle "note_on" event
                if len(event_type.split("_")) > 2:
                    current_pitch = int(event_type.split("_")[2])
                else:
                    # If there's no pitch value (just "note_on_"), skip to next token
                    continue
                
                if last_note_value is None:
                    # First note - use as reference
                    last_note_value = current_pitch
                    event = f"note_on_{current_pitch}"
                else:
                    # Maintain relative pitch movement
                    pitch_diff = current_pitch - last_note_value
                    current_pitch = (last_note_value + pitch_diff) % 128  # Wrap around within MIDI pitch range
                    last_note_value = current_pitch
                    event = f"note_on_{current_pitch}"
                
                # Store the active note to generate "note_off" later
                active_notes[current_pitch] = event

            elif event_type.startswith("note_off_"):
                # Handle "note_off" event for ending the note
                if len(event_type.split("_")) > 2:
                    pitch = int(event_type.split("_")[2])
                    if pitch in active_notes:
                        event = f"note_off_{pitch}"
                        del active_notes[pitch]  # Remove from active notes after it's turned off
        
        elif event_type.startswith("time_"):
            # Handle time shift or tempo change
            if event_type == "time_shift":
                time_shift += 1  # Increment time shift (could be adjusted based on model output)
                event = f"time_shift_{time_shift}"
            elif event_type.startswith("tempo_"):
                # Adjust tempo if it's a tempo event
                new_tempo = int(event_type.split("_")[1])
                tempo = new_tempo
                event = f"tempo_{tempo}"
        
        else:
            event = event_type  # Other events (like modifiers, etc.)

        generated.append(next_token)

        # Convert the predicted token to one-hot encoding
        next_one_hot = torch.zeros(1, vocab_size, device=device)
        next_one_hot[0, next_token] = 1

        # Update sequences
        current_src = torch.cat([current_src, next_one_hot], dim=0)[-seq_len:, :]
        current_tgt = torch.cat([current_tgt, next_one_hot], dim=0)[-seq_len:, :]

    # Return the generated sequence
    return generated[seq_len:]
