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