import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """

    # (x - mean) / sqrt (var^2  + eps)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    A = ( x - mean ) / np.sqrt(var + eps) 

    return gamma * A + beta
    
    

    