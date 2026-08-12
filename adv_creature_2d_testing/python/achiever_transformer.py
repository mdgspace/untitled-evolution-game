import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class AchieverTransformer(nn.Module):
    """
    Achiever Transformer as specified in §4 of Creature Brain Architecture.
    
    Processes variable-cardinality token sets T = {(c_i, v_i)} where:
      - c_i: joint identity index (0 is reserved for goal token, 1..catalog_size-1 for joint types)
      - v_i: scalar/vector value associated with token i (joint angle, goal vector, or joint action)
      
    Outputs a flat linear projection of the final transformer hidden state.
    """
    def __init__(
        self,
        catalog_size: int = 64,
        val_dim: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        out_dim: int = 1
    ):
        super().__init__()
        self.catalog_size = catalog_size
        self.val_dim = val_dim
        self.d_model = d_model
        
        # Identity embedding W_c
        self.c_embed = nn.Embedding(catalog_size, d_model)
        # Value projection W_v
        self.v_proj = nn.Linear(val_dim, d_model)
        
        # Transformer Layer stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        self.per_token_readout = nn.Linear(d_model, 1)
        self.global_readout = nn.Linear(d_model, out_dim)

    def forward(
        self,
        cat_ids: torch.Tensor,   # Shape: [batch_size, num_tokens] (long)
        values: torch.Tensor,    # Shape: [batch_size, num_tokens, val_dim] (float)
        mask: Optional[torch.Tensor] = None, # Shape: [batch_size, num_tokens] (bool, True for padding)
        mode: str = "per_token" # "per_token" for M2 action output, "global" for M3 position delta
    ) -> torch.Tensor:
        """
        Forward pass for set processing.
        """
        # Embed token categories and project token values
        # z_i^(0) = W_c * c_i + W_v * v_i
        z_c = self.c_embed(cat_ids) # [B, N, d_model]
        z_v = self.v_proj(values)   # [B, N, d_model]
        z = z_c + z_v              # [B, N, d_model]

        # Transformer encoder processing
        # mask is key_padding_mask: True where padding token
        z_out = self.transformer_encoder(z, src_key_padding_mask=mask) # [B, N, d_model]

        if mode == "per_token":
            # For M2: output per joint action delta
            out = self.per_token_readout(z_out) # [B, N, out_dim]
            return out
        elif mode == "global":
            # For M3: aggregate hidden representations (e.g. mean pooling over non-masked tokens)
            if mask is not None:
                valid_mask = (~mask).unsqueeze(-1).float() # [B, N, 1]
                sum_z = (z_out * valid_mask).sum(dim=1)
                count = valid_mask.sum(dim=1).clamp(min=1.0)
                pooled = sum_z / count
            else:
                pooled = z_out.mean(dim=1) # [B, d_model]
            
            out = self.global_readout(pooled) # [B, out_dim]
            return out
        else:
            raise ValueError(f"Unknown readout mode: {mode}")
