import os
import torch
import pickle
import pretty_midi

def create_midi_from_events(events, output_path="output.mid", default_tempo=120):
    """
    Convert a sequence of MIDI events to a MIDI file.

    :param events: List of MIDI events (e.g., note_on, note_off, time_shift, tempo changes).
    :param output_path: Path to save the generated MIDI file.
    :param default_tempo: The default tempo to apply to the MIDI file (in BPM).
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=default_tempo)
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program)

    current_time = 0
    active_notes = {}  # Dictionary to track active notes by pitch

    for event in events:
        try:
            if event.startswith("time_shift_"):
                # Extract time shift duration and convert to seconds
                duration = int(event.split("_")[2]) / 1000.0  # Convert milliseconds to seconds
                current_time += duration

            elif event.startswith("note_on_"):
                # Extract pitch and create a new note
                pitch = int(event.split("_")[2])
                velocity = 100  # Default velocity
                active_notes[pitch] = pretty_midi.Note(
                    velocity=velocity, pitch=pitch, start=current_time, end=current_time + 0.5
                )  # Default end time is 0.5 seconds after start

            elif event.startswith("note_off_"):
                # Extract pitch and end the corresponding note
                pitch = int(event.split("_")[2])
                if pitch in active_notes:
                    note = active_notes.pop(pitch)
                    note.end = current_time  # Update the end time of the note
                    piano.notes.append(note)

            elif event.startswith("tempo_"):
                # Adjust the tempo if a tempo change event occurs
                new_tempo = int(event.split("_")[1])
                midi.adjust_tempo(new_tempo, current_time)

            else:
                print(f"Unknown event: {event}")

        except (ValueError, IndexError) as e:
            print(f"Invalid event format: {event} ({e})")

    # Handle any active notes without a note_off event
    for note in active_notes.values():
        note.end = current_time
        piano.notes.append(note)

    # Add the piano instrument to the MIDI object
    midi.instruments.append(piano)

    # Write to the output MIDI file
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
