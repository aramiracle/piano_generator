import os
import pretty_midi
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

def midi_to_event_sequence(midi_path):
    """
    Converts a MIDI file into a sequence of note-on, note-off, and time-shift events.
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    events = []
    prev_time = 0

    for note in sorted(midi_data.instruments[0].notes, key=lambda n: n.start):
        # Add time-shift events
        time_shift = int((note.start - prev_time) * 1000)  # Convert to milliseconds
        if time_shift > 0:
            events.append(f"time_shift_{time_shift}")
        prev_time = note.start

        # Add note-on and note-off events
        events.append(f"note_on_{note.pitch}")
        events.append(f"note_off_{note.pitch}")

    return events

def preprocess_dataset(dataset_path, output_path):
    """
    Preprocesses the Maestro dataset by converting MIDI files to event sequences.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    sequences = []
    midi_files = list(dataset_path.rglob("*.midi"))

    # Use tqdm to track progress
    for midi_file in tqdm(midi_files, desc="Processing MIDI files", unit="file"):
        try:
            events = midi_to_event_sequence(midi_file)
            sequences.append(events)
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
    
    # Save using pickle
    with open(output_path / "maestro_sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)
    print(f"Preprocessed dataset saved to {output_path / 'maestro_sequences.pkl'}")

def main():
    # Path to Maestro dataset and output
    DATASET_PATH = "dataset/maestro-v3.0.0"
    OUTPUT_PATH = "dataset/preprocessed"

    preprocess_dataset(DATASET_PATH, OUTPUT_PATH)

if __name__=="__main__":
    main()
