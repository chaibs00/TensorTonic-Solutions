import torch
import torch.nn as nn

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Returns an embedding layer with the requested dimensions.
    """
    # Create a matrix ( row = vocab_size , col = d_model )
    matrix = nn.Embedding(vocab_size, d_model)

    # Fill it with weights : 
    nn.init.normal_(matrix.weight, mean=0.0, std=1.0)
    
    return matrix

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Returns scaled token embeddings.
    """
    
    return embedding(tokens)*math.sqrt(embedding.embedding_dim)