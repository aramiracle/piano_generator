import os
import torch
import pickle
import pretty_midi

def create_midi_from_events(events, output_path="output.mid"):
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program)

    # Parse events and create notes
    current_time = 0
    for event in events:
        if event.startswith("note_on_"):
            # Extract pitch and set velocity
            pitch = int(event.split("_")[2])
            velocity = 100  # Set a fixed velocity
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=current_time, end=current_time + 0.5)
            piano.notes.append(note)
        elif event.startswith("note_off_"):
            # In pretty_midi, note-offs are not explicitly needed; duration is handled by `end`.
            pass
        elif event.startswith("time_shift_"):
            # Adjust current time by the specified amount
            time_shift = float(event.split("_")[2]) / 100  # Assuming time_shift is in 100ms units
            current_time += time_shift

    # Add the piano instrument to the PrettyMIDI object
    midi.instruments.append(piano)

    # Write to a MIDI file
    midi.write(output_path)
    print(f"MIDI file created: {output_path}")

def generate_square_subsequent_mask(size, device):
    mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
    return ~mask

def load_vocab_size(preprocessed_path):
    with open(preprocessed_path, 'rb') as f:
        data = pickle.load(f)
    return len(data['vocab'])

def load_latest_checkpoint(model, optimizer, checkpoint_dir="checkpoints"):
    """
    Load the latest checkpoint to resume training.

    Args:
        model (nn.Module): Model to load the checkpoint into.
        optimizer (torch.optim.Optimizer): Optimizer to load the checkpoint state into.
        checkpoint_dir (str): Directory containing checkpoint files.

    Returns:
        model (nn.Module): The model loaded with the latest checkpoint.
        optimizer (torch.optim.Optimizer): The optimizer loaded with the checkpoint state.
        epoch (int): The epoch number from the checkpoint.
        loss (float): The loss from the checkpoint.
    """
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoint_files:
        print("No checkpoints found. Starting from scratch.")
        return model, optimizer, 0, 0.0
    
    # Sort checkpoints by epoch number (assumes filename format 'checkpoint_epoch_X.pt')
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = checkpoint_files[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from epoch {epoch}, loss {loss:.4f}")
    return model, optimizer, epoch, loss

def test_model(model, dataloader, reverse_vocab, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients needed during inference
        # Loop through the batches in the dataloader
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass through the model
            output = model(
                src=inputs,
                tgt=targets[:, :-1],  # Input to the model is the target sequence excluding the last token
                src_mask=None,  # Assuming no source mask is needed (or you can pass generate_square_subsequent_mask if required)
                tgt_mask=None,  # Assuming no target mask is needed (or you can pass generate_square_subsequent_mask if required)
                src_key_padding_mask=None,  # Assuming no padding mask (or pass one if needed)
                tgt_key_padding_mask=None,  # Assuming no padding mask (or pass one if needed)
            )  # Output shape: (batch_size, seq_len, vocab_size)

            # Get the predicted token indices (last token for each batch)
            predicted_note_idx = output[:, -1, :].argmax(dim=-1)  # Shape: (batch_size,) 

            # Get the real token (last token in the target sequence)
            real_note_idx = targets[:, -1, :].argmax(dim=-1)  # Shape: (batch_size,)

            # Loop over each example in the batch
            for i in range(inputs.size(0)):
                # Assuming reverse_vocab maps indices to notes
                predicted_note = reverse_vocab[predicted_note_idx[i].item()]  # Convert predicted index to note
                real_note = reverse_vocab[real_note_idx[i].item()]  # Convert real index to note

                # Output the predicted and real notes for this example
                print(f"Predicted note: {predicted_note}")
                print(f"Real note: {real_note}")
