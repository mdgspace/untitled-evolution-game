import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class AchieverTransformer(nn.Module):
    """
    Achiever Transformer — §4 of Creature Brain Architecture.

    Processes variable-cardinality token sets T = {(c_i, v_i)} where:
      - c_i: joint identity index into a fixed catalog (long).
              Index 0 is the reserved CLS/goal token.
              Indices 1..catalog_size-1 are joint types.
      - v_i: float vector of dimension val_dim (joint angle+sensors, goal, or action).

    Two readout modes:
      "per_token" — used by M2 (action-selector).
                    Outputs one scalar per non-CLS token: the Δjoint angle.
      "global"    — used by M3 (dynamics predictor).
                    A learnable [CLS] token is prepended; its final hidden
                    state is projected linearly to out_dim (3D position delta).
                    This matches the spec's  W_out · vec(Z^(L)) + b_out  intent
                    while remaining seq-len-agnostic.

    Token embedding:   z_i^(0) = W_c · c_i + W_v · v_i          (§4, eq. 1)
    Transformer stack: standard pre-norm encoder layers             (§4, eq. 2)
    M2 readout:        per_token_readout(z_i^(L))  for i ≥ 1      (§4, eq. 3)
    M3 readout:        global_readout(z_0^(L))  where z_0 is CLS  (§4, eq. 3)
    """

    def __init__(
        self,
        catalog_size: int = 64,
        val_dim: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        out_dim: int = 1,
    ):
        super().__init__()
        self.catalog_size = catalog_size
        self.val_dim = val_dim
        self.d_model = d_model

        # W_c: identity embedding  (includes slot 0 = CLS/goal token)
        self.c_embed = nn.Embedding(catalog_size, d_model)
        # W_v: value projection
        self.v_proj = nn.Linear(val_dim, d_model)

        # Learnable CLS value vector (used only in "global" / M3 mode).
        # The CLS token always gets identity 0; its *value* is this parameter
        # rather than a physics measurement, so M3 has a dedicated aggregation
        # anchor whose representation is freely learned.
        self.cls_value = nn.Parameter(torch.zeros(1, 1, val_dim))

        # Pre-norm Transformer encoder (norm_first=True matches §4's LN-before-attn)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # M2 readout: per-token scalar  →  Δjoint angle
        self.per_token_readout = nn.Linear(d_model, 1)
        # M3 readout: CLS hidden state  →  3-D position delta (or out_dim)
        self.global_readout = nn.Linear(d_model, out_dim)

    def forward(
        self,
        cat_ids: torch.Tensor,           # [B, N]  long
        values: torch.Tensor,            # [B, N, val_dim]  float
        mask: Optional[torch.Tensor] = None,  # [B, N] bool, True = padding
        mode: str = "per_token",
    ) -> torch.Tensor:
        """
        Forward pass.

        "per_token" (M2):
            Processes T_2 = {(c_0, g_t)} ∪ {(c_i, q_t^(i))}.
            Returns [B, N, 1] — one scalar per token (goal token output
            is at position 0 and should be sliced away by the caller).

        "global" (M3):
            Prepends a CLS token (id=0, learned value) to T_3.
            Returns [B, out_dim] — the CLS hidden state after the encoder,
            linearly projected.  The padding mask is extended by one False
            column for the CLS position so it is never masked out.
        """
        B, N = cat_ids.shape

        if mode == "global":
            # --- prepend CLS token ---
            cls_id = torch.zeros(B, 1, dtype=torch.long, device=cat_ids.device)
            cls_val = self.cls_value.expand(B, 1, self.val_dim)  # [B, 1, val_dim]

            cat_ids_in = torch.cat([cls_id, cat_ids], dim=1)    # [B, N+1]
            values_in  = torch.cat([cls_val, values],  dim=1)   # [B, N+1, val_dim]

            if mask is not None:
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
                mask_in  = torch.cat([cls_mask, mask], dim=1)   # [B, N+1]
            else:
                mask_in = None
        else:
            cat_ids_in = cat_ids
            values_in  = values
            mask_in    = mask

        # z_i^(0) = W_c · c_i + W_v · v_i
        z_c = self.c_embed(cat_ids_in)      # [B, N(+1), d_model]
        z_v = self.v_proj(values_in)        # [B, N(+1), d_model]
        z   = z_c + z_v                     # [B, N(+1), d_model]

        # Transformer encoder stack (§4, eq. 2)
        z_out = self.transformer_encoder(z, src_key_padding_mask=mask_in)

        if mode == "per_token":
            # M2: project every token's hidden state to a scalar action.
            # Caller slices z_out[:, 1:, :] to strip the goal token if desired,
            # but here we project all N tokens so shapes are consistent.
            return self.per_token_readout(z_out)   # [B, N, 1]

        elif mode == "global":
            # M3: use only the CLS token's final hidden state (position 0
            # after prepending).  This is the spec's W_out · vec(Z^(L)) + b_out
            # equivalent for variable-length sequences.
            cls_hidden = z_out[:, 0, :]            # [B, d_model]
            return self.global_readout(cls_hidden)  # [B, out_dim]

        else:
            raise ValueError(f"Unknown readout mode: {mode!r}. Expected 'per_token' or 'global'.")
