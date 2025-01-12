import pretty_midi
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

def midi_to_event_sequence(midi_path, max_events=512):
    """
    Converts a MIDI file into a sequence of note-on, note-off, and time-shift events.
    Uses musical timing (480 ticks per quarter note).
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    events = []
    prev_time = 0
    TICKS_PER_QUARTER = 480  # Standard MIDI resolution

    # Get tempo changes
    tempo = 120  # Default tempo (BPM)
    if midi_data.get_tempo_changes()[1].size > 0:
        tempo = midi_data.get_tempo_changes()[1][0]
    
    seconds_per_tick = 60.0 / (tempo * TICKS_PER_QUARTER)

    # Process notes into events
    for note in sorted(midi_data.instruments[0].notes, key=lambda n: n.start):
        # Convert time difference to ticks
        time_diff_seconds = note.start - prev_time
        time_diff_ticks = int(time_diff_seconds / seconds_per_tick)

        # Quantize time shifts to common musical divisions
        if time_diff_ticks > 0:
            # Quantize to common note lengths:
            # 480 = quarter note
            # 240 = eighth note
            # 120 = sixteenth note
            # 60 = thirty-second note
            quantized_ticks = min(480, max(60, round(time_diff_ticks / 60) * 60))
            events.append(f"time_shift_{quantized_ticks}")
        prev_time = note.start

        # Add note events with velocity information
        velocity = int(note.velocity)
        events.append(f"note_on_{note.pitch}_{velocity}")
        
        # Calculate note duration in ticks
        duration_ticks = int((note.end - note.start) / seconds_per_tick)
        quantized_duration = min(1920, max(60, round(duration_ticks / 60) * 60))  # Max 4 quarter notes
        events.append(f"note_off_{note.pitch}_{quantized_duration}")

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
    Ensures special tokens are at the beginning.
    """
    # Start with special tokens
    special_tokens = ["pad", "sos", "eos"]
    unique_events = set()
    
    # Collect all unique events
    for seq in sequences:
        unique_events.update(seq)
    
    # Remove special tokens from unique events if they exist
    unique_events = unique_events - set(special_tokens)
    
    # Create vocabulary mapping
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    next_idx = len(special_tokens)
    
    # Add remaining tokens
    for event in sorted(unique_events):
        vocab[event] = next_idx
        next_idx += 1
        
    return vocab

def events_to_indices(events, vocab):
    """
    Converts event sequences to index sequences using the vocabulary.
    Adds start-of-sequence and end-of-sequence tokens.
    """
    return [vocab["sos"]] + [vocab[event] for event in events] + [vocab["eos"]]

def preprocess_dataset(dataset_path, output_path, max_events=512, max_files=None):
    """
    Preprocesses the Maestro dataset by converting MIDI files to paired sequences.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    midi_files = list(dataset_path.rglob("*.midi")) + list(dataset_path.rglob("*.mid"))
    if max_files:
        midi_files = midi_files[:max_files]

    src_sequences, tgt_sequences = [], []
    for midi_file in tqdm(midi_files, desc="Processing MIDI files", unit="file"):
        try:
            # Source sequence
            src_events = midi_to_event_sequence(midi_file, max_events)

            # Create target sequence (transposed up by one semitone)
            tgt_events = []
            for event in src_events:
                if event.startswith("note_on_") or event.startswith("note_off_"):
                    parts = event.split("_")
                    pitch = int(parts[2])
                    if len(parts) > 3:  # Has velocity/duration
                        tgt_events.append(f"{parts[0]}_{parts[1]}_{pitch + 1}_{parts[3]}")
                    else:
                        tgt_events.append(f"{parts[0]}_{parts[1]}_{pitch + 1}")
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