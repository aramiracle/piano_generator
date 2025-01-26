import os
import torch
from tqdm import tqdm
from utils import generate_square_subsequent_mask

def train_model(model, dataloader, optimizer, criterion, device, predict_length, one_hot=True, epochs=10, start_epoch=0):
    """
    Train a Transformer model to predict the last predict_length tokens of the target sequence.
    The model predicts the entire sequence of length predict_length, and loss is calculated
    across all predictions.
    """
    model.train()
    
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for (src, tgt) in progress_bar:
            src = src.to(device)
            tgt = tgt.to(device)
            
            batch_size = src.size(0)
            
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

            # Forward pass for initial sequence
            output = model(
                src=src,
                tgt=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            
            # Initialize tensors to store predictions and losses
            all_predictions = []
            total_loss = 0
            
            # Get the last output from the initial sequence
            current_input = tgt_input.clone()
            
            # Predict each token in the sequence
            for i in range(predict_length):
                # Get the model's prediction
                pred = output[:, -1]  # Get last token prediction [batch_size, vocab_size]
                all_predictions.append(pred)
                
                # Calculate loss for this position
                if one_hot:
                    current_target = tgt_output[:, i].float()  # [batch_size, vocab_size]
                    position_loss = criterion(pred, current_target)
                else:
                    current_target = tgt_output[:, i].long()  # [batch_size]
                    position_loss = criterion(pred, current_target)
                
                total_loss += position_loss
                
                # Update input for next prediction
                if i < predict_length - 1:
                    if one_hot:
                        # Convert prediction to one-hot
                        pred_probs = torch.softmax(pred, dim=-1)
                        prediction = torch.zeros_like(tgt[:, 0])
                        prediction.scatter_(-1, pred_probs.argmax(dim=-1, keepdim=True), 1)
                    else:
                        # Get token indices
                        prediction = pred.argmax(dim=-1)
                    
                    current_input = torch.cat([current_input[:, 1:], prediction.unsqueeze(1)], dim=1)
                    
                    # Update target mask for new sequence length
                    tgt_mask = generate_square_subsequent_mask(current_input.size(1), device)
                    
                    # Generate new key padding mask if needed
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
            loss = total_loss / predict_length
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            
            # Optimizer step
            optimizer.step()
            
            # Stack predictions and calculate accuracy
            predictions = torch.stack(all_predictions, dim=1)  # [batch_size, predict_length, vocab_size]
            
            if one_hot:
                _, predicted = torch.max(predictions, dim=-1)  # [batch_size, predict_length]
                correct = (predicted == torch.argmax(tgt_output, dim=-1)).sum().item()
                total = predicted.numel()
            else:
                _, predicted = torch.max(predictions, dim=-1)  # [batch_size, predict_length]
                correct = (predicted == tgt_output).sum().item()
                total = predicted.numel()
            
            # Update metrics
            loss_item = loss.item()
            epoch_loss += loss_item
            epoch_correct += correct
            epoch_total += total
            
            # Update progress bar
            accuracy = 100 * correct / total
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