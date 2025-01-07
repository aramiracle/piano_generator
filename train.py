import os
import torch
from utils import generate_square_subsequent_mask
from tqdm import tqdm

def train_model(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    
    # Ensure checkpoints directory exists
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            # Input validation
            if torch.max(src) >= model.embedding.num_embeddings or torch.min(src) < 0:
                print(f"Invalid src indices found in batch {batch_idx}")
                print(f"Max src value: {torch.max(src).item()}")
                print(f"Min src value: {torch.min(src).item()}")
                print(f"Vocab size: {model.embedding.num_embeddings}")
                continue
            
            # Move data to device
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Prepare target input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Generate masks
            src_mask = generate_square_subsequent_mask(src.size(1), device)
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device)
            
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            output = model(src, tgt_input, src_mask, tgt_mask)
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Update metrics
            loss_item = loss.item()
            epoch_loss += loss_item
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss_item:.4f}'})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint in the checkpoints directory
        checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
