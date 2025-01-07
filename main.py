import torch
import torch.nn as nn
from model import TransformerModel
from train import train_model
from dataset import get_dataloader
from generate import generate_sequence

def main():
    # Hyperparameters
    VOCAB_SIZE = 512  # Adjust based on events
    EMBED_SIZE = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    FF_DIM = 2048
    MAX_LEN = 512
    DROPOUT = 0.1
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Preprocessed Data
    SEQUENCES_PATH = "dataset/preprocessed/maestro_sequences.pkl"
    dataloader = get_dataloader(SEQUENCES_PATH, batch_size=BATCH_SIZE)

    # Initialize Model
    model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, FF_DIM, MAX_LEN, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Train Model
    train_model(model, dataloader, optimizer, criterion, device, epochs=EPOCHS)

    # Generate Sequence
    start_seq = [0]  # Example start sequence
    generated_seq = generate_sequence(model, start_seq, max_length=100, vocab_size=VOCAB_SIZE, device=device)
    print("Generated Sequence:", generated_seq)

if __name__ == "__main__":
    main()
