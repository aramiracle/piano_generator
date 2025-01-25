import os
import torch
import torch.nn as nn
from model import TransformerModel, TransformerOneHotModel
from train import train_model
from dataset import get_dataloader
from generate import generate_sequence_from_batch
from utils import load_vocab_size, create_midi_from_events, load_latest_checkpoint

def main():
    # Load preprocessed data first to get vocabulary size
    SEQUENCES_PATH = "dataset/preprocessed/maestro_paired_sequences.pkl"
    VOCAB_SIZE = load_vocab_size(SEQUENCES_PATH)
    
    # Updated Hyperparameters
    EMBED_SIZE = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    FF_DIM = 512
    MAX_LEN = 2048  # Adjusted based on typical sequence lengths
    DROPOUT = 0.2
    EPOCHS = 30  # Increased for potentially better convergence
    BATCH_SIZE = 64  # Reduced if memory is a concern
    GEN_SEQ_LEN = 1000
    LABEL_SMOOTHING = 0.3
    LEARNING_RATE = 3e-4  # Adjusted for quicker optimization with AdamW
    PREDICT_LENGTH = 10
    OUTPUT_FILENAME = "generated_song.mid"
    CHECKPOINT_DIR = "checkpoints"
    ONE_HOT = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Preprocessed Data
    dataloader, _, reverse_vocab = get_dataloader(SEQUENCES_PATH, batch_size=BATCH_SIZE, one_hot=ONE_HOT)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # Initialize Model
    if ONE_HOT:
        model = TransformerOneHotModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, FF_DIM, MAX_LEN, DROPOUT).to(device)
    else:
        model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, FF_DIM, MAX_LEN, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Load model from the latest checkpoint if available
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model, optimizer, start_epoch, _ = load_latest_checkpoint(model, optimizer, checkpoint_dir=CHECKPOINT_DIR)

    # Train Model (start from the next epoch)
    train_model(model, dataloader, optimizer, criterion, device, predict_length=PREDICT_LENGTH, one_hot=ONE_HOT, epochs=EPOCHS, start_epoch=start_epoch)
    
    # Generate Sequence
    generated_sequence = generate_sequence_from_batch(
        model=model,
        dataloader=dataloader,
        gen_seq_len=GEN_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        device=device,
        reverse_vocab=reverse_vocab,
        one_hot=ONE_HOT
    )

    # Convert generated indices back to events
    generated_events = [reverse_vocab[idx] for idx in generated_sequence]
    print("Generated Sequence:", generated_events)

    # Create MIDI from generated events
    create_midi_from_events(generated_events, output_path=OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
