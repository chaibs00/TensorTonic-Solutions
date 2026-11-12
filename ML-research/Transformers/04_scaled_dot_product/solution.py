import torch
import numpy as np 

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """

    # Q (batch, q_len, d_k)
    # K (batch, k_len, d_k)
    # V (batch, k_len, d_v)
    # Ouput (batch, q_len, d_v)

    # Get d_k 
    _,_,d_k = K.shape
    
    # Compute scores :
    # (-2 , -1) => (second to last dimention )
    # S (batch, q_length, k_length)
    S = (Q @ K.transpose(-2, -1) ) / np.sqrt(d_k)

    # Apply softmax : 
    Att = torch.softmax(S, dim=-1)

    # output : 
    # Att(batch, q_length, k_length ) @ V(batch, k_len, d_v) = (batch; q_length, d_v)
    output = Att @ V 

    return output


    