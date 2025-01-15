import os
import torch
from tqdm import tqdm
from utils import generate_square_subsequent_mask

def train_model(model, dataloader, optimizer, criterion, device, epochs=10, start_epoch=0):
    """
    Train a Transformer model.

    Args:
        model (nn.Module): Transformer model.
        dataloader (DataLoader): DataLoader providing training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use for training (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        start_epoch (int): Epoch to start training from (useful when resuming training).
    """
    model.train()
    
    # Ensure checkpoints directory exists
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for (src, tgt) in progress_bar:
            # Move data to device
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Prepare target input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Generate masks
            src_seq_len = src.size(1)
            tgt_seq_len = tgt_input.size(1)
            src_mask = generate_square_subsequent_mask(src_seq_len, device)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
            
            # Key padding masks
            src_key_padding_mask = (src.sum(dim=-1) == 0)
            tgt_key_padding_mask = (tgt_input.sum(dim=-1) == 0)

            # Clear gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            output = model(
                src=src,
                tgt=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            
            # Reshape for loss calculation
            output = output.contiguous().view(-1, model.vocab_size)  # Model output, already float
            tgt_output = tgt_output.contiguous().view(-1, model.vocab_size).float()  # Target, ensure it's integer for class indices

            # Calculate loss
            loss = criterion(output, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            
            # Optimizer step
            optimizer.step()
            
            # Update metrics
            loss_item = loss.item()
            epoch_loss += loss_item

            # Calculate accuracy
            _, predicted = torch.max(output, dim=-1)  # Get the index of the max log-probability
            correct = (predicted == torch.argmax(tgt_output, dim=-1).view(-1)).sum().item()  # Compare with ground truth
            epoch_correct += correct
            epoch_total += tgt_output.numel()  # Total number of tokens

            # Update progress bar
            accuracy = 100 * epoch_correct / epoch_total
            progress_bar.set_postfix({'loss': f'{loss_item:.4f}', 'accuracy': f'{accuracy:.2f}%'})

        avg_loss = epoch_loss / len(dataloader)
        avg_accuracy = 100 * epoch_correct / epoch_total
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.2f}%")
        
        # Save checkpoint in the checkpoints directory
        checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'accuracy': avg_accuracy,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
