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
            time_shift = float(event.split("_")[2]) / 1000  # Assuming time_shift is in 1 second units
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
