import numpy as np
import math

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    A = (x - mean) / np.sqrt( var + eps)

    output = gamma * A + beta 

    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
    """
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
    Att = softmax(S)

    # output
    # (batch, num_heads, q_len, d_k)
    out_heads = np.matmul(Att, V_heads)

    # cancat output :
    # we need (batch, q_len, num_heads, d_k) so to get d_model
    out_concat = out_heads.transpose(0, 2, 1, 3).reshape(batch, q_len, d_model)

    return np.matmul(out_concat, W_o)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    # x W1 + b1 
    hidden = np.dot(x, W1) + b1
    
    # Apply ReLU 
    hidden = np.maximum(0, hidden)

    # Relu(x W1 + b1) W2 + b2
    output = np.dot(hidden, W2) + b2

    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """

    # 1. Apply multi-head Attention 
    Att = multi_head_attention(
        x, x, x, 
        W_q, W_k, W_v,
        W_o, num_heads  
    )

    # 2. LN
    x = layer_norm(
        x + Att, gamma1, beta1
    )

    # 3. FFN
    ffn = feed_forward(
        x, W1, b1, W2, b2
    )

    # 4. LN
    x = layer_norm(
        x + ffn , gamma2, beta2
    )

    return x 
    

    
