import numpy as np
import math

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """

    # Each postion => Row 
    # Each dimension => column

    arr = np.ones((seq_length, d_model))

    # Create pos (seq_length, 1)

    # Iterate though positions : 
    for pos in range(seq_length) :
        # Iterate though dimentions : 
        for i in range(d_model):
            # (i//2) each sin-cos pair get same frequency 
            S = pos / ( 10000 ** (2*(i//2) / d_model))
            # even :
            if (i % 2 == 0 ) :
                arr[pos][i] = math.sin(S)
            else : 
                arr[pos][i] =  math.cos(S)

    # Return (seq_length, d_model)

    return arr 

    