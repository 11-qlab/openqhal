"""
Transformer policy for the RL atom rearrangement compiler.

Handles variable grid sizes. Each grid cell becomes a token with
features (grid, target, diff, row/col embedding). Attention over
all cells produces a global embedding, which feeds two heads:

  - policy: logits over the action space
  - value:  scalar V(s)

Action space layout (default for grid size G):
  axis (2) x index (G) x direction (2) x magnitude (2) = 8*G actions.
Can be overridden with an explicit n_actions to match an environment.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        max_grid_size: int = 8,
        n_cell_features: int = 5,   # grid, target, diff, row/H, col/W
        hidden: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        n_actions: int = None,      # explicit override; else 8*max_grid_size
    ):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.hidden = hidden

        self.n_axes = 2
        self.n_dirs = 2
        self.n_mags = 2
        if n_actions is not None:
            self.n_actions = n_actions
        else:
            self.n_actions = self.n_axes * max_grid_size * self.n_dirs * self.n_mags

        # Project cell features to hidden
        self.embed = nn.Linear(n_cell_features, hidden)

        # Positional encoding for up to max_grid_size^2 cells
        max_cells = max_grid_size * max_grid_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_cells, hidden) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 2,
            batch_first=True, dropout=0.0, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=n_layers)

        # Progress projection
        self.progress_proj = nn.Linear(1, hidden)

        # Heads
        self.policy_head = nn.Linear(hidden, self.n_actions)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, grid, target, diff, progress,
                H: int = None, W: int = None):
        """
        grid, target, diff: (B, H, W) tensors
        progress: (B,) or (B, 1) tensor
        """
        B, Hh, Ww = grid.shape
        if H is None: H = Hh
        if W is None: W = Ww

        rows = torch.arange(H, device=grid.device).float() / max(H, 1)
        cols = torch.arange(W, device=grid.device).float() / max(W, 1)
        row_feat = rows.view(1, H, 1).expand(B, H, W)
        col_feat = cols.view(1, 1, W).expand(B, H, W)

        cells = torch.stack([
            grid.float(), target.float(), diff.float(),
            row_feat, col_feat,
        ], dim=-1)                                       # (B, H, W, 5)
        cells = cells.view(B, H * W, -1)                 # (B, HW, 5)

        x = self.embed(cells)
        x = x + self.pos_embed[:, :H * W, :]

        if progress.dim() == 1:
            progress = progress.unsqueeze(-1)
        prog_tok = self.progress_proj(progress).unsqueeze(1)
        x = torch.cat([prog_tok, x], dim=1)              # (B, 1+HW, hidden)

        h = self.encoder(x)
        global_emb = h[:, 1:, :].mean(dim=1)             # (B, hidden)

        logits = self.policy_head(global_emb)            # (B, n_actions)
        value = self.value_head(global_emb).squeeze(-1)  # (B,)
        return logits, value


class TransformerAdapter(nn.Module):
    """
    Wraps a TransformerPolicy and accepts the flat observation the
    environment produces. Reconstructs (grid, target, diff, progress)
    and calls the underlying Transformer.

    Compatible with the existing PPO / REINFORCE trainers.
    """
    def __init__(self, H: int, W: int, max_grid_size: int = 8,
                 hidden: int = 64, n_layers: int = 2, n_heads: int = 4,
                 n_actions: int = None):
        super().__init__()
        self.H = H
        self.W = W
        self.grid_size = H * W
        if n_actions is None:
            n_actions = 2 * max(H, W) * 2 * 2
        self.trunk = TransformerPolicy(
            max_grid_size=max_grid_size,
            hidden=hidden, n_layers=n_layers, n_heads=n_heads,
            n_actions=n_actions)
        self.n_actions = self.trunk.n_actions
        self.state_dim = 3 * self.grid_size + 1

    def forward(self, x):
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        B = x.shape[0]
        gs = self.grid_size
        grid = x[:, :gs].view(B, self.H, self.W)
        target = x[:, gs:2 * gs].view(B, self.H, self.W)
        diff = x[:, 2 * gs:3 * gs].view(B, self.H, self.W)
        progress = x[:, 3 * gs]
        logits, value = self.trunk(grid, target, diff, progress,
                                    self.H, self.W)
        if squeeze:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value
