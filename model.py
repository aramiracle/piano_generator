import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """
    A Transformer model for sequence-to-sequence tasks.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimensionality of the model embeddings.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        num_layers (int): Number of encoder and decoder layers.
        ff_dim (int): Dimensionality of the feedforward network in each layer.
        max_len (int): Maximum sequence length for positional encoding.
        dropout (float): Dropout rate applied throughout the model.
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ff_dim, max_len=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        self.input_projection = nn.Linear(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_len, d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def _generate_positional_encoding(max_len, d_model):
        """
        Generates positional encoding for the model.
        Args:
            max_len (int): Maximum sequence length.
            d_model (int): Dimensionality of the embeddings.
        Returns:
            Tensor: Positional encoding tensor of shape (1, max_len, d_model).
        """
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:d_model//2])
        return pe

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the Transformer model.
        Args:
            src (Tensor): Source sequence of shape (batch_size, seq_len, vocab_size).
            tgt (Tensor): Target sequence of shape (batch_size, seq_len, vocab_size).
            src_mask, tgt_mask, memory_mask: Attention masks for source, target, and memory.
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask: Padding masks.
        Returns:
            Tensor: Output logits of shape (batch_size, seq_len, vocab_size).
        """
        src_emb = self._embed(src)
        tgt_emb = self._embed(tgt)
        
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        return self.fc_out(out)
    
    def _embed(self, x):
        # Ensure x is of type float
        x = x.float()  # Convert to float if not already
        x_emb = self.input_projection(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=x.device))
        
        # Add positional encoding by indexing
        seq_len = x.size(1)  # Get the sequence length of the input
        x_emb = x_emb + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        return self.dropout(x_emb)


