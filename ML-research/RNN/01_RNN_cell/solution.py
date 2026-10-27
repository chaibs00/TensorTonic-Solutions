import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, W_xh: np.ndarray,
             W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Returns the float64 next hidden state.
    """
    
    # Pay attention to the shapes !!

    # (D * (D,H)) + (H * (H,H)) + (H,) => (H,)
    func = (x_t @ W_xh.T) + (h_prev @ W_hh.T) + b_h 

    h_t = np.tanh(func).astype(np.float64)
    
    return h_t

    