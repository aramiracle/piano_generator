import torch
import torch.nn as nn
from model import TransformerModel
from train import train_model
from dataset import get_dataloader
from generate import generate_sequence
from utils import load_vocab_size, create_midi_from_events

def main():
    # Load preprocessed data first to get vocabulary size
    SEQUENCES_PATH = "dataset/preprocessed/maestro_sequences.pkl"
    VOCAB_SIZE = load_vocab_size(SEQUENCES_PATH)
    
    # Hyperparameters
    EMBED_SIZE = 256
    NUM_HEADS = 4
    NUM_LAYERS = 2
    FF_DIM = 1024
    MAX_LEN = 512  # Match with preprocessing max_events
    DROPOUT = 0.1
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Preprocessed Data
    dataloader, vocab, reverse_vocab = get_dataloader(SEQUENCES_PATH, batch_size=BATCH_SIZE)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # Initialize Model
    model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, FF_DIM, MAX_LEN, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['pad'])  # Ignore padding tokens

    # Train Model
    train_model(model, dataloader, optimizer, criterion, device, epochs=EPOCHS)

    # Generate Sequence
    start_tokens = ["note_on_60"]  # Start with middle C
    start_seq = [vocab[token] for token in start_tokens]
    generated_seq = generate_sequence(model, start_seq, max_length=100, vocab_size=VOCAB_SIZE, device=device)
    
    # Convert generated indices back to events
    generated_events = [reverse_vocab[idx] for idx in generated_seq]
    print("Generated Sequence:", generated_events)

    # Create MIDI from generated events
    midi_path = "generated_sequence.mid"
    create_midi_from_events(generated_events, output_path=midi_path)

if __name__ == "__main__":
    main()