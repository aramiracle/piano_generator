import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
from collections import defaultdict

class MIDIDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing MIDI event sequences.
    """
    def __init__(self, sequences_path):
        """
        Initialize the dataset by loading sequences from a pickle file.

        Args:
            sequences_path (str or Path): Path to the pickle file containing preprocessed MIDI sequences.
        """
        # Load sequences
        with open(sequences_path, "rb") as f:
            self.sequences = pickle.load(f)

        # Ensure the sequences are a list of lists
        if not isinstance(self.sequences, list) or not all(isinstance(seq, list) for seq in self.sequences):
            raise ValueError("Loaded sequences must be a list of lists.")

        # Create a mapping from event string to unique integer
        self.event_to_idx = self._create_event_mapping(self.sequences)

    def _create_event_mapping(self, sequences):
        """
        Create a mapping from event string to integer index.

        Args:
            sequences (list): A list of event sequences.
        
        Returns:
            dict: A dictionary mapping event strings to integers.
        """
        event_to_idx = defaultdict(lambda: len(event_to_idx))
        
        # Add start and end tokens
        event_to_idx['<start>'] = 0
        event_to_idx['<end>'] = 1
        
        # Iterate through all sequences and add all event strings to the mapping
        for seq in sequences:
            for event in seq:
                event_to_idx[event]  # Access to populate defaultdict
                
        return dict(event_to_idx)

    def __len__(self):
        """Return the total number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieve a single sequence and prepare source (src) and target (tgt).

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            tuple: A tuple (src, tgt), where `src` is the input sequence and `tgt` is the target sequence.
        """
        seq = self.sequences[idx]
        src = [self.event_to_idx.get(event, self.event_to_idx['<start>']) for event in seq[:-1]]
        tgt = [self.event_to_idx.get(event, self.event_to_idx['<start>']) for event in seq[1:]]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch):
    """
    Custom collate function for padding variable-length sequences in a batch.

    Args:
        batch (list): A list of tuples (src, tgt) from the dataset.

    Returns:
        tuple: Padded source and target tensors.
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

def get_dataloader(sequences_path, batch_size=64, shuffle=True):
    """
    Create a DataLoader for the MIDI dataset.

    Args:
        sequences_path (str or Path): Path to the pickle file containing preprocessed MIDI sequences.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: A PyTorch DataLoader for the MIDI dataset.
    """
    dataset = MIDIDataset(sequences_path)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
