import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """

    # x W1 + b1 
    hidden = np.dot(x, W1) + b1
    
    # Apply ReLU 
    hidden = np.maximum(0, hidden)

    # Relu(x W1 + b1) W2 + b2
    output = np.dot(hidden, W2) + b2

    return output