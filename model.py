import torch
import torch.nn as nn

class TransformerOneHotModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ff_dim, max_len=512, dropout=0.1):
        super(TransformerOneHotModel, self).__init__()
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
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:d_model//2])
        return pe

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
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
        x = x.float()
        x_emb = self.input_projection(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=x.device))
        seq_len = x.size(1)
        x_emb = x_emb + self.positional_encoding[:, :seq_len, :].to(x.device)
        return self.dropout(x_emb)
    
    def generate_sequence(self, src, predict_length, src_mask=None, tgt_mask=None, memory_mask=None, 
                        src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, temperature=1.0):
        """
        Generate a sequence of predictions with one-hot encoding.
        
        Args:
            src (Tensor): Source sequence of shape (batch_size, seq_len, vocab_size)
            predict_length (int): Length of the sequence to generate
            src_mask (Tensor, optional): Source sequence mask
            tgt_mask (Tensor, optional): Target sequence mask
            memory_mask (Tensor, optional): Memory mask
            src_key_padding_mask (Tensor, optional): Source key padding mask
            tgt_key_padding_mask (Tensor, optional): Target key padding mask
            memory_key_padding_mask (Tensor, optional): Memory key padding mask
            temperature (float): Sampling temperature (higher = more random)
        
        Returns:
            Tensor: Generated sequence of shape (batch_size, predict_length, vocab_size)
        """
        device = src.device
        batch_size = src.size(0)

        # Initialize target sequence with zeros (use appropriate start token for specific tasks)
        tgt = torch.zeros((batch_size, 1, self.vocab_size), device=device)
        generated_sequence = []

        for _ in range(predict_length):
            # Create masks for the current sequence
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Get predictions
            with torch.no_grad():
                output = self.forward(
                    src=src,
                    tgt=tgt,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
            
            next_token_logits = output[:, -1:, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs.squeeze(1), 1).unsqueeze(1)
            
            # Convert to one-hot
            next_token_onehot = torch.zeros((batch_size, 1, self.vocab_size), device=device)
            next_token_onehot.scatter_(-1, next_token.unsqueeze(-1), 1)
            
            generated_sequence.append(next_token_onehot)
            tgt = torch.cat([tgt, next_token_onehot], dim=1)

        return torch.cat(generated_sequence, dim=1)
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ff_dim, max_len=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        self.embedding = nn.Embedding(vocab_size, d_model)
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
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:d_model//2])
        return pe

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
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
        x_emb = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=x.device))
        seq_len = x.size(1)
        x_emb = x_emb + self.positional_encoding[:, :seq_len, :].to(x.device)
        return self.dropout(x_emb)
    
    def generate_sequence(self, src, predict_length, src_mask=None, tgt_mask=None, memory_mask=None, 
                        src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, temperature=1.0):
        """
        Generate a sequence of predictions with token indices.
        
        Args:
            src (Tensor): Source sequence of token indices of shape (batch_size, seq_len)
            predict_length (int): Length of the sequence to generate
            src_mask (Tensor, optional): Source sequence mask
            tgt_mask (Tensor, optional): Target sequence mask
            memory_mask (Tensor, optional): Memory mask
            src_key_padding_mask (Tensor, optional): Source key padding mask
            tgt_key_padding_mask (Tensor, optional): Target key padding mask
            memory_key_padding_mask (Tensor, optional): Memory key padding mask
            temperature (float): Sampling temperature (higher = more random)
        
        Returns:
            Tensor: Generated sequence of token indices of shape (batch_size, predict_length)
        """
        device = src.device
        batch_size = src.size(0)

        # Initialize target sequence with first token (you might want to use a special start token)
        tgt = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        generated_sequence = []

        for _ in range(predict_length):
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Get predictions
            with torch.no_grad():
                output = self.forward(
                    src=src,
                    tgt=tgt,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
            
            next_token_logits = output[:, -1:, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs.squeeze(1), 1)
            
            generated_sequence.append(next_token)
            tgt = torch.cat([tgt, next_token], dim=1)

        return torch.cat(generated_sequence, dim=1)
