import os
import torch
from tqdm import tqdm
from utils import generate_square_subsequent_mask

def train_model(model, dataloader, optimizer, criterion, device, predict_length, one_hot=True, epochs=10, start_epoch=0):
    """
    Memory-efficient version of the training function that processes predictions
    one at a time and clears unnecessary tensors.
    """
    model.train()
    
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src = src.to(device)
            tgt = tgt.to(device)
            
            if one_hot:
                tgt_input = tgt[:, :-predict_length]
                tgt_output = tgt[:, -predict_length:]
                src_key_padding_mask = (src.sum(dim=-1) == 0)
                tgt_key_padding_mask = (tgt_input.sum(dim=-1) == 0)
            else:
                tgt_input = tgt[:, :-predict_length]
                tgt_output = tgt[:, -predict_length:]
                src_key_padding_mask = (src == 0)
                tgt_key_padding_mask = (tgt_input == 0)
            
            src_seq_len = src.size(1)
            tgt_seq_len = tgt_input.size(1)
            src_mask = generate_square_subsequent_mask(src_seq_len, device)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
            
            optimizer.zero_grad(set_to_none=True)

            # Initial forward pass
            current_input = tgt_input
            output = model(
                src=src,
                tgt=current_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            
            batch_loss = 0
            batch_correct = 0
            batch_total = 0
            
            # Process one prediction at a time
            for i in range(predict_length):
                # Get prediction for current position
                pred = output[:, -1]  # [batch_size, vocab_size]
                
                # Calculate loss
                if one_hot:
                    current_target = tgt_output[:, i].float()
                    step_loss = criterion(pred, current_target)
                else:
                    current_target = tgt_output[:, i].long()
                    step_loss = criterion(pred, current_target)
                
                batch_loss += step_loss
                
                # Calculate accuracy
                if one_hot:
                    _, predicted = torch.max(pred, dim=-1)
                    correct = (predicted == torch.argmax(current_target, dim=-1)).sum().item()
                else:
                    _, predicted = torch.max(pred, dim=-1)
                    correct = (predicted == current_target).sum().item()
                
                batch_correct += correct
                batch_total += predicted.numel()
                
                # Update input for next prediction if needed
                if i < predict_length - 1:
                    with torch.no_grad():
                        if one_hot:
                            prediction = torch.zeros_like(tgt[:, 0])
                            prediction.scatter_(-1, predicted.unsqueeze(-1), 1)
                        else:
                            prediction = predicted
                        
                        # Update input sequence
                        current_input = torch.cat([current_input[:, 1:], prediction.unsqueeze(1)], dim=1)
                        
                        # Update mask for new sequence length
                        tgt_mask = generate_square_subsequent_mask(current_input.size(1), device)
                        
                        # Update padding mask
                        if one_hot:
                            tgt_key_padding_mask = (current_input.sum(dim=-1) == 0)
                        else:
                            tgt_key_padding_mask = (current_input == 0)
                        
                        # Get next prediction
                        output = model(
                            src=src,
                            tgt=current_input,
                            src_mask=src_mask,
                            tgt_mask=tgt_mask,
                            src_key_padding_mask=src_key_padding_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                        )
            
            # Average loss over sequence length
            loss = batch_loss / predict_length
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            
            # Optimizer step
            optimizer.step()
            
            # Update metrics
            loss_item = loss.item()
            epoch_loss += loss_item
            epoch_correct += batch_correct
            epoch_total += batch_total
            
            # Update progress bar
            accuracy = 100 * batch_correct / batch_total
            progress_bar.set_postfix({'loss': f'{loss_item:.4f}', 'accuracy': f'{accuracy:.2f}%'})

        avg_loss = epoch_loss / len(dataloader)
        avg_accuracy = 100 * epoch_correct / epoch_total
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.2f}%")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'accuracy': avg_accuracy,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")