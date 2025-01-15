import os
import torch
import pickle
import pretty_midi

def create_midi_from_events(events, output_path="output.mid", tempo=120):
    """
    Convert a sequence of MIDI events to a MIDI file.
    
    :param events: List of MIDI events (e.g., note_on, note_off, time_shift, tempo changes).
    :param output_path: Path to save the generated MIDI file.
    :param tempo: The tempo to apply to the MIDI file (in BPM).
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)  # Set tempo
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program)

    # Constants for timing
    BEAT_LENGTH = 0.5  # Length of a quarter note in seconds at 120 BPM
    current_time = 0
    active_notes = {}  # Dictionary to keep track of active notes

    for event in events:
        if event.startswith("note_on_"):
            pitch = int(event.split("_")[2])
            velocity = 64  # Medium velocity for more natural sound
            active_notes[pitch] = {
                'start': current_time,
                'velocity': velocity
            }
            
        elif event.startswith("note_off_"):
            pitch = int(event.split("_")[2])
            if pitch in active_notes:
                # Create note with actual duration
                note = pretty_midi.Note(
                    velocity=active_notes[pitch]['velocity'],
                    pitch=pitch,
                    start=active_notes[pitch]['start'],
                    end=current_time
                )
                piano.notes.append(note)
                del active_notes[pitch]
                
        elif event.startswith("time_shift_"):
            duration = int(event.split("_")[2])
            # Convert duration to musical time
            # duration 480 = quarter note, 240 = eighth note, etc.
            time_shift = (duration / 480) * BEAT_LENGTH
            current_time += time_shift

        elif event.startswith("tempo_"):
            # Adjust tempo based on the event (e.g., "tempo_100" for 100 BPM)
            tempo = int(event.split("_")[1])
            midi.get_end_time()  # Force tempo adjustment, even if not explicitly used

    # Handle any notes that haven't received a note_off event
    for pitch, note_data in active_notes.items():
        note = pretty_midi.Note(
            velocity=note_data['velocity'],
            pitch=pitch,
            start=note_data['start'],
            end=current_time
        )
        piano.notes.append(note)

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
