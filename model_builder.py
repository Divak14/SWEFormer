import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt

from scipy.io import loadmat
import numpy as np
from typing import Tuple
from matplotlib.colors import ListedColormap
import pandas as pd

from torch import nn, Tensor

from typing import Optional, Any

from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
import torch.nn as nn
from torch.nn import functional as F

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal = False):
        # Pass through the multihead attention
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask, need_weights=True, average_attn_weights=False)
        
        # Store attention weights in the module itself
        self.attn_weights = attn_weights
        
        # Apply the rest of the TransformerEncoderLayer operations
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class SWEFormer(nn.Module):
    def __init__(self, feat_dim=4, max_len=170+60, d_model=16, n_heads=4, num_layers=3, dim_feedforward=64, num_classes=1,
                 num_categories=12, cat_embed_dim=8, water_frac_dim = 1, elev_dim = 1, dropout=0.1, pos_encoding='learnable', activation='relu', norm='BatchNorm', freeze=False, output_concat_switch = True, combined_continous_emb = 16):
        super(SWEFormer, self).__init__()

        self.max_len = max_len
        self.d_model = d_model - cat_embed_dim - water_frac_dim -elev_dim
        self.n_heads = n_heads
        
        self.output_concat_switch = output_concat_switch

        # Linear projection layer for the input features
        self.project_inp = nn.Linear(feat_dim, d_model)

        # Positional encoding
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        # Transformer encoder with custom layers
        encoder_layer = CustomTransformerEncoderLayer(d_model+combined_continous_emb, n_heads, dim_feedforward, dropout * (1.0 - freeze), activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Activation function
        self.act = _get_activation_fn(activation)

        # Dropout layer
        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.cat_embed_dim = cat_embed_dim
        self.water_frac_dim = water_frac_dim
        self.elev_dim = elev_dim

        # Embedding layer for the categorical variable
        self.cat_embed = nn.Embedding(num_categories, cat_embed_dim)

        # Output layer
        if output_concat_switch:
            self.output_layer = self.build_output_module(d_model+combined_continous_emb, max_len, num_classes, cat_embed_dim, water_frac_dim, elev_dim )
            
        else:
            self.output_layer = self.build_output_module(d_model+combined_continous_emb, max_len, num_classes, cat_embed_dim = 0, water_frac_dim= 0, elev_dim = 0 )

        # List to store attention weights for visualization
        self.attention_weights = []

        # Linear layer for continuous input (optional, depending on dimensionality)
        self.project_continuous = nn.Linear(cat_embed_dim+2, combined_continous_emb)  # Assuming continuous input is scalar

    def build_output_module(self, d_model, max_len, num_classes, cat_embed_dim, water_frac_dim, elev_dim ):
        # Add a ReLU activation after the linear layer
        output_layer = nn.Sequential(
            nn.Linear(d_model * max_len+cat_embed_dim+water_frac_dim + elev_dim, num_classes),
            nn.ReLU()
        )
        return output_layer

    def forward(self, X, categorical_input, continuous_input, elevation_input, padding_masks, return_attention=False):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            categorical_input: (batch_size,) tensor containing the categorical input for each sample
            continuous_input: (batch_size, 1) tensor containing the continuous variable for each sample
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # Process the categorical input through the embedding layer
        cat_embed = self.cat_embed(categorical_input)  # (batch_size, cat_embed_dim)

        # Project continuous input if needed (for dimensionality alignment, if not scalar)
        continuous_input = continuous_input.unsqueeze(-1)  # Ensure continuous_input has shape (batch_size, 1)
        elevation_input = elevation_input.unsqueeze(-1)
        
        combined_continuous = self.project_continuous(torch.cat((cat_embed, continuous_input, elevation_input), dim=-1))
        
        # Permute because PyTorch convention for transformers is [seq_length, batch_size, feat_dim].
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # Project input vectors to d_model dimensional space
        inp1 = self.pos_enc(inp)  # Add positional encoding

        
        combined_continuous_expanded = combined_continuous.unsqueeze(0).expand(inp1.size(0), -1, -1) 
        # Concatenate the time series input, categorical embedding, and continuous variable along the feature dimension
        combined_inp = torch.cat((inp1, combined_continuous_expanded), dim=-1)  # (seq_length, batch_size, d_model + cat_embed_dim + 1)

        # Clear previous attention weights
        self.attention_weights = []

        # Transformer encoder forward pass
        output1 = self.transformer_encoder(combined_inp, src_key_padding_mask=padding_masks)

        # Capture the attention weights from each layer
        for layer in self.transformer_encoder.layers:
            self.attention_weights.append(layer.attn_weights)

        # Apply activation function
        output2 = self.act(output1)  # (seq_length, batch_size, d_model)
        output3 = output2.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        #output4 = self.dropout1(output3)
        output4 = output3

        # Zero-out padding embeddings
        output5 = output4 * ~padding_masks.unsqueeze(-1)  # (batch_size, seq_length, d_model)
        output6 = output5.reshape(output5.shape[0], -1)  # (batch_size, seq_length * d_model)
        if self.output_concat_switch:
            output_concat = torch.cat((output6, combined_continuous_expanded),dim=-1)
        else:
            output_concat = output6
        # Final output layer
        output = self.output_layer(output_concat)  # (batch_size, num_classes)

        # If return_attention is True, return both output and output1
        if return_attention:
            return torch.squeeze(output), output1
        else:
            return torch.squeeze(output)