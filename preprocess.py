import pretty_midi
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

def midi_to_event_sequence(midi_path, max_events=512, quantized_shift_gap=50, max_time_shift=500):
    """
    Converts a MIDI file into a sequence of events with reduced vocabulary size.
    Handles note-on/note-off events and incorporates time differences as integers.
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    events = []
    prev_time = 0
    active_notes = set()  # Tracks currently active notes for "note_off" events

    # Process notes into events
    for note in sorted(midi_data.instruments[0].notes, key=lambda n: n.start):
        # Add time-shift events in terms of integer time difference (milliseconds)
        time_shift = int((note.start - prev_time) * 1000)  # Convert time shift to milliseconds (integer)
        if time_shift > 0:
            # Quantizing time shift to steps (integer) and ensuring it's at most 500ms
            quantized_shift = min((time_shift // quantized_shift_gap) * quantized_shift_gap, max_time_shift)
            events.append(f"time_shift_{quantized_shift}")
        prev_time = note.start

        # Add note-on event (start of the note)
        events.append(f"note_on_{note.pitch}")
        active_notes.add(note.pitch)  # Mark the note as active
        
        # Add note-off event (end of the note)
        events.append(f"note_off_{note.pitch}")

        # Truncate if sequence gets too long
        if len(events) >= max_events:
            events = events[:max_events]
            break

    # Pad sequence if it's too short
    if len(events) < max_events:
        events.extend(["pad"] * (max_events - len(events)))

    return events

def create_vocabulary(sequences):
    """
    Creates a vocabulary mapping from events to indices.
    """
    unique_events = set()
    for seq in sequences:
        unique_events.update(seq)
    vocab = {event: idx for idx, event in enumerate(sorted(unique_events))}
    return vocab

def events_to_indices(events, vocab):
    """
    Converts event sequences to index sequences using the vocabulary.
    """
    return [vocab[event] for event in events]

def indices_to_events(indices, reverse_vocab):
    """
    Converts index sequences back to event sequences using the reverse vocabulary.
    """
    return [reverse_vocab[idx] for idx in indices]

def generate_midi_from_events(events, output_path):
    """
    Generates a MIDI file from the sequence of events and saves it to the specified path.
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Piano
    prev_time = 0
    active_notes = {}

    for event in events:
        if event.startswith("time_shift_"):
            try:
                time_shift = int(event.split("_")[2]) / 1000  # Convert back to seconds
                prev_time += time_shift
            except (ValueError, IndexError):
                print(f"Invalid time_shift event: {event}")
                continue
        elif event.startswith("note_on_"):
            try:
                pitch = int(event.split("_")[2])
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=prev_time, end=prev_time + 0.5)  # 0.5 sec duration
                instrument.notes.append(note)
                active_notes[pitch] = note
            except (ValueError, IndexError):
                print(f"Invalid note_on event: {event}")
                continue
        elif event.startswith("note_off_"):
            try:
                pitch = int(event.split("_")[2])
                if pitch in active_notes:
                    note = active_notes.pop(pitch)
                    note.end = prev_time  # End time is the current time
            except (ValueError, IndexError):
                print(f"Invalid note_off event: {event}")
                continue
        else:
            print(f"Unknown event: {event}")

    midi.instruments.append(instrument)
    midi.write(str(output_path))  # Convert Path to string here

def preprocess_dataset(dataset_path, output_path, max_events=512, max_files=None):
    """
    Preprocesses the Maestro dataset with reduced vocabulary size and handles note-on/off and time shift events.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    midi_files = list(dataset_path.rglob("*.midi"))
    if max_files:
        midi_files = midi_files[:max_files]

    src_sequences, tgt_sequences = [], []
    for midi_file in tqdm(midi_files, desc="Processing MIDI files", unit="file"):
        try:
            # Source sequence
            src_events = midi_to_event_sequence(midi_file, max_events)

            # Simple transformation for target sequence (shift up by 1)
            tgt_events = []
            for event in src_events:
                if event.startswith("note_on_"):
                    pitch = int(event.split("_")[2])
                    tgt_events.append(f"note_on_{(pitch + 1) % 88}")  # Wrap around within range
                elif event.startswith("note_off_"):
                    pitch = int(event.split("_")[2])
                    tgt_events.append(f"note_off_{(pitch + 1) % 88}")  # Wrap around within range
                else:
                    tgt_events.append(event)

            src_sequences.append(src_events)
            tgt_sequences.append(tgt_events)
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
            continue

    # Create vocabulary
    vocab = create_vocabulary(src_sequences + tgt_sequences)

    # Convert sequences to indices
    src_indices = [events_to_indices(seq, vocab) for seq in src_sequences]
    tgt_indices = [events_to_indices(seq, vocab) for seq in tgt_sequences]

    # Convert to numpy arrays
    src_indices = np.array(src_indices, dtype=np.int32)
    tgt_indices = np.array(tgt_indices, dtype=np.int32)

    # Save data
    with open(output_path / "maestro_paired_sequences.pkl", "wb") as f:
        pickle.dump({
            'src_sequences': src_indices,
            'tgt_sequences': tgt_indices,
            'vocab': vocab,
            'reverse_vocab': {idx: event for event, idx in vocab.items()}
        }, f)
    
    print(f"Preprocessed dataset saved to {output_path / 'maestro_paired_sequences.pkl'}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sequence length: {max_events}")
    print(f"Number of sequences: {len(src_sequences)}")

    # Save one sample MIDI file from the source sequences for checking
    sample_midi_file = output_path / "sample_output.mid"
    generate_midi_from_events(src_sequences[0], sample_midi_file)
    print(f"Sample MIDI file saved to {sample_midi_file}")

def main():
    DATASET_PATH = "dataset/maestro-v3.0.0"
    OUTPUT_PATH = "dataset/preprocessed"
    MAX_EVENTS = 1568
    MAX_FILES = None
    
    preprocess_dataset(DATASET_PATH, OUTPUT_PATH, max_events=MAX_EVENTS, max_files=MAX_FILES)

if __name__ == "__main__":
    main()
