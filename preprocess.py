import pretty_midi
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

def midi_to_event_sequence(midi_path, max_events=512, min_note=21, max_note=108, quantized_shift_gap=50):
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
        # Ensure valid note pitch is within the specified range
        mapped_pitch = min(max_note, max(min_note, note.pitch))

        # Add time-shift events in terms of integer time difference (milliseconds)
        time_shift = int((note.start - prev_time) * 1000)  # Convert time shift to milliseconds (integer)
        if time_shift > 0:
            # Quantizing time shift to 25ms steps (integer)
            quantized_shift = (time_shift // quantized_shift_gap) * quantized_shift_gap
            events.append(f"time_shift_{quantized_shift}")
        prev_time = note.start

        # Add note-on event (start of the note)
        relative_pitch = mapped_pitch - min_note
        events.append(f"note_on_{relative_pitch}")
        active_notes.add(relative_pitch)  # Mark the note as active
        
        # Add note-off event (end of the note)
        events.append(f"note_off_{relative_pitch}")

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

def main():
    DATASET_PATH = "dataset/maestro-v3.0.0"
    OUTPUT_PATH = "dataset/preprocessed"
    MAX_EVENTS = 768
    MAX_FILES = None
    
    preprocess_dataset(DATASET_PATH, OUTPUT_PATH, max_events=MAX_EVENTS, max_files=MAX_FILES)

if __name__ == "__main__":
    main()
