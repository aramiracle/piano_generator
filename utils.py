import torch

def generate_square_subsequent_mask(size):
    """
    Generate a square mask for the transformer that ensures each position can only attend to previous positions.

    Args:
        size (int): The size of the mask (sequence length).

    Returns:
        Tensor: The generated mask with shape (size, size).
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)  # Upper triangular matrix with 1s above the diagonal
    return mask == 0  # Return the inverse: 0s for valid positions and 1s for invalid positions
