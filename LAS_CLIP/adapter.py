import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2) # (B, C, H, W)

class ContextAdapter(nn.Module):
    """Gated Token-Mask Interaction adapter (Proposal 5).
    
    Generates per-head, per-row attention bias matrix (B, H, N, N) via
    gated dot-product between token-conditioned gate vectors and mask features.
    
    Architecture:
        - Mask encoder: Conv2d patchify -> LayerNorm2d -> GELU -> (B, N, h)
        - Token encoder: Linear D -> h on detached tokens -> (B, N, h)
        - Gate generator: Linear h -> H*h + sigmoid -> (B, H, N, h)
        - Bias: G @ F_mask^T -> (B, H, N, N)
        - Learnable per-head per-layer temperature (zero-init)
    """

    def __init__(self, input_size=224, patch_size=16,
                 d_model=768, num_heads=12, hidden_dim=64,
                 mask_layers=3, dropout_rate=0.2):
        super().__init__()
        self.grid_size = input_size // patch_size
        self.n_patches = self.grid_size ** 2
        self.n_tokens = self.n_patches + 1
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mask_layers = mask_layers
        self.dropout_rate = dropout_rate

        # Mask encoder: patchify binary mask -> spatial feature
        self.mask_branch = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=patch_size, stride=patch_size),
            LayerNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
        )

        # Learned CLS mask feature prepended to patch mask features
        self.cls_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.cls_mask_token, std=0.02)

        # Token encoder: project visual tokens down to hidden_dim
        self.token_proj = nn.Linear(d_model, hidden_dim)

        # Layer conditioning: additive embedding per layer
        self.layer_embed = nn.Embedding(mask_layers, hidden_dim)
        nn.init.zeros_(self.layer_embed.weight)

        # Gate generator: per-head gate vectors from token features
        self.gate_proj = nn.Linear(hidden_dim, num_heads * hidden_dim)
        self.gate_dropout = nn.Dropout(dropout_rate)

        # Learnable per-head, per-layer temperature (zero-init for stable start)
        self.layer_temp = nn.Parameter(torch.zeros(mask_layers, num_heads))

    def get_mask_feat(self, mask):
        mf = self.mask_branch(mask)           # (B, h, G, G)
        B = mf.size(0)
        mf_flat = mf.view(B, self.hidden_dim, -1).permute(0, 2, 1)  # (B, P, h)
        cls_mf = self.cls_mask_token.expand(B, -1).unsqueeze(1)      # (B, 1, h)
        return torch.cat([cls_mf, mf_flat], dim=1)        # (B, N, h)

    def forward(self, mask, x, layer_idx=0, mask_feat=None):
        """
        Args:
            mask: (B, 1, H, W) binary/continuous mask
            x: token sequence from ViT layer.
               Shape (L, B, D) [sequence first] or (B, L, D) [batch first].
            layer_idx: index within the mask_layers range (0-based).
            mask_feat: precomputed mask features.
            
        Returns:
            bias: (B, H, N, N) attention bias matrix.
        """
        if mask_feat is None:
            mask_feat = self.get_mask_feat(mask)

        B = mask.size(0) if mask is not None else x.size(1)

        # Extract CLS and Patch tokens, auto-detect layout
        if x.size(1) == B:        # Sequence first (L, B, D)
            tokens = x.permute(1, 0, 2)   # (B, L, D)
        elif x.size(0) == B:      # Batch first (B, L, D)
            tokens = x
        else:                     # Fallback: treat as sequence first
            tokens = x.permute(1, 0, 2)

        tokens = tokens.detach().to(dtype=self.token_proj.weight.dtype)

        # Token encoder
        f_tok = self.token_proj(tokens)   # (B, N, h)

        # Layer conditioning
        l_emb = self.layer_embed.weight[layer_idx]  # (h,)
        f_tok = f_tok + l_emb

        # Gate: per-head gate vectors with sigmoid activation
        gate = self.gate_proj(f_tok)                              # (B, N, H*h)
        gate = gate.view(B, self.n_tokens, self.num_heads, self.hidden_dim)
        gate = gate.permute(0, 2, 1, 3)                          # (B, H, N, h)
        gate = torch.sigmoid(gate)
        gate = self.gate_dropout(gate)

        # Bias via gated interaction: G @ F_mask^T
        f_mask = mask_feat.unsqueeze(1)                           # (B, 1, N, h)
        bias = torch.matmul(gate, f_mask.transpose(-1, -2))       # (B, H, N, N)

        # Per-head, per-layer temperature scaling (direct scaling for zero-init)
        temp = self.layer_temp[layer_idx]                         # (H,)
        bias = bias * temp.view(1, self.num_heads, 1, 1)

        return bias


def save_adapter(adapter: ContextAdapter, path: str,
                 extra_meta: Optional[Dict[str, Any]] = None):
    """Save adapter weights and config to a checkpoint file.
    
    Args:
        adapter: ContextAdapter instance.
        path: file path (.pt) to save.
        extra_meta: optional dict of extra metadata (e.g. training info).
    """
    config = {
        "grid_size": adapter.grid_size,
        "n_patches": adapter.n_patches,
        "num_heads": adapter.num_heads,
        "hidden_dim": adapter.hidden_dim,
        "mask_layers": adapter.mask_layers,
        "dropout_rate": adapter.dropout_rate,
    }
    # Reverse-derive input_size and patch_size from grid_size
    # grid_size = input_size // patch_size, and Conv2d kernel = patch_size
    conv_weight = adapter.mask_branch[0].weight  # (hidden_dim, 1, ps, ps)
    patch_size = conv_weight.shape[-1]
    input_size = adapter.grid_size * patch_size
    d_model = adapter.token_proj.in_features

    config["input_size"] = input_size
    config["patch_size"] = patch_size
    config["d_model"] = d_model

    payload = {
        "config": config,
        "state_dict": adapter.state_dict(),
    }
    if extra_meta:
        payload["meta"] = extra_meta

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_adapter(path: str, device: str = "cpu",
                 strict: bool = True) -> ContextAdapter:
    """Load adapter from a checkpoint file.
    
    Args:
        path: file path (.pt) to load.
        device: target device for the loaded adapter.
        strict: whether to enforce strict state_dict matching.
        
    Returns:
        Loaded ContextAdapter instance on the specified device.
    """
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]

    adapter = ContextAdapter(
        input_size=cfg["input_size"],
        patch_size=cfg["patch_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        hidden_dim=cfg["hidden_dim"],
        mask_layers=cfg["mask_layers"],
        dropout_rate=cfg["dropout_rate"],
    )
    adapter.load_state_dict(ckpt["state_dict"], strict=strict)
    adapter.to(device)
    return adapter
