import pretty_midi
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

def midi_to_event_sequence(midi_path, max_events=512, min_note=21, max_note=108):
    """
    Converts a MIDI file into a sequence of events with reduced vocabulary size:
    - Limits note range (default: piano range from A0 to C8)
    - More aggressive time quantization (25ms steps)
    - Combines note-on/off into single events
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    events = []
    prev_time = 0

    # Process notes into events
    for note in sorted(midi_data.instruments[0].notes, key=lambda n: n.start):
        # Skip notes outside the specified range
        if note.pitch < min_note or note.pitch > max_note:
            continue

        # Add time-shift events
        time_shift = int((note.start - prev_time) * 1000)  # Convert to milliseconds
        if time_shift > 0:
            # More aggressive quantization (25ms steps)
            quantized_shift = min(1000, max(25, time_shift // 25 * 25))
            events.append(f"time_{quantized_shift}")
        prev_time = note.start

        # Simplified note events (normalized to C4 = 60)
        relative_pitch = note.pitch - min_note
        events.append(f"note_{relative_pitch}")

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
    Preprocesses the Maestro dataset with reduced vocabulary size.
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
                if event.startswith("note_"):
                    pitch = int(event.split("_")[1])
                    tgt_events.append(f"note_{(pitch + 1) % 88}")  # Wrap around within range
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