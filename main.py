import torch
import torch.nn as nn
from model import TransformerModel
from train import train_model
from dataset import get_dataloader
from generate import generate_sequence
from utils import load_vocab_size, create_midi_from_events, load_latest_checkpoint, test_model

def main():
    # Load preprocessed data first to get vocabulary size
    SEQUENCES_PATH = "dataset/preprocessed/maestro_paired_sequences.pkl"
    VOCAB_SIZE = load_vocab_size(SEQUENCES_PATH)
    
    # Updated Hyperparameters
    EMBED_SIZE = 128
    NUM_HEADS = 4
    NUM_LAYERS = 3
    FF_DIM = 1024
    MAX_LEN = 512  # Adjusted based on typical sequence lengths
    DROPOUT = 0.1
    EPOCHS = 200  # Increased for potentially better convergence
    BATCH_SIZE = 32  # Reduced if memory is a concern
    START_SEQ_SIZE = 5
    MAX_GEN_LEN = 10
    LEARNING_RATE = 3e-4  # Adjusted for quicker optimization with AdamW

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Preprocessed Data
    dataloader, _, reverse_vocab = get_dataloader(SEQUENCES_PATH, batch_size=BATCH_SIZE)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # Initialize Model
    model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, FF_DIM, MAX_LEN, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Load model from the latest checkpoint if available
    model, optimizer, start_epoch, _ = load_latest_checkpoint(model, optimizer)

    # Train Model (start from the next epoch)
    train_model(model, dataloader, optimizer, criterion, device, epochs=EPOCHS, start_epoch=start_epoch)

    # Testing model
    test_model(model, dataloader, reverse_vocab, device)
    
    # Generate Sequence
    generated_seq = generate_sequence(model, START_SEQ_SIZE, MAX_GEN_LEN, VOCAB_SIZE, device)
    
    # Convert generated indices back to events
    generated_events = [reverse_vocab[idx] for idx in generated_seq]
    print("Generated Sequence:", generated_events)

    # Create MIDI from generated events
    midi_path = "generated_sequence.mid"
    create_midi_from_events(generated_events, output_path=midi_path)

if __name__ == "__main__":
    main()

