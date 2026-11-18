import numpy as np
import torch
import math

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """


    # ===========================================================================

    # Q (batch, q_len, d_model)
    # K (batch, k_len, d_model)
    # V (batch, v_len, d_model)

    batch, q_len, d_model = Q.shape
    _, k_len, _ = K.shape
    _, v_len, _ = V.shape


    # get d_k to divise by d_k
    d_k = d_model // num_heads

    # Projection : all (batch, seq_len, d_model)
    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    # d_model =  num_heads * d_k
    # Reshaping (batch, seq_len, nums_heads, d_k) => (batch, num_head, seq_len, d_k  ) 
    Q_heads = Q_proj.reshape(batch, q_len, num_heads, d_k).transpose(0, 2, 1,3)
    K_heads = K_proj.reshape(batch, k_len, num_heads, d_k).transpose(0, 2, 1,3)
    V_heads = V_proj.reshape(batch, v_len, num_heads, d_k).transpose(0, 2, 1,3)


    # Scores :
    # Q_heads (batch, nums_heads, q_len, d_k) 
    # k_heads(batch, nums_heads, k_len, d_k)
    # K_heads.T(batch, nums_heads, d_k, k_len)
    S = np.matmul( Q_heads , K_heads.transpose(0,1,3,2)) / math.sqrt(d_k)


    # Apply softmax :
    # (batch,  num_heads, )
    exp = np.exp(S - np.max(S, axis=-1, keepdims=True))
    Att = exp / np.sum(exp, axis=-1, keepdims=True)

    # output
    # (batch, num_heads, q_len, d_k)
    out_heads = np.matmul(Att, V_heads)

    # cancat output :
    # we need (batch, q_len, num_heads, d_k) so to get d_model
    out_concat = out_heads.transpose(0, 2, 1, 3).reshape(batch, q_len, d_model)



    # fINAL 
    return np.matmul(out_concat, W_o)

    

    
    
    
    

    

    

    