"""The value network.

It looks at one position from one player's side, right after that player's
guess, and predicts that player's final duel score (RULES.md W5).

Input:
- the answers that still fit the player's colors (each answer word has its own
  learned list of numbers, and the network takes their average), and
- 11 extra numbers: guesses used so far (one-hot, 0 to 6), whether the player
  went first, and the size of the answer list.
"""

import math

import numpy as np
import torch
import torch.nn as nn

N_ANSWERS = 2315
N_EXTRA = 11


def extras(n_left, used, first):
    """The 11 extra numbers for one position."""
    e = np.zeros(N_EXTRA, dtype=np.float32)
    e[used] = 1.0
    e[7] = 1.0 if first else 0.0
    e[8] = math.log2(n_left + 1) / 12.0
    e[9] = 1.0 if n_left == 1 else 0.0
    e[10] = 1.0 if n_left == 2 else 0.0
    return e


class ValueNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.words = nn.EmbeddingBag(N_ANSWERS, hidden, mode="mean")
        self.body = nn.Sequential(
            nn.Linear(hidden + N_EXTRA, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.words.weight, std=0.1)

    def forward(self, indices, offsets, extra):
        h = self.words(indices, offsets)
        return self.body(torch.cat([h, extra], dim=1)).squeeze(1)


def batch_tensors(member_lists, extra_rows, device):
    """Pack many answer lists into the flat form EmbeddingBag needs."""
    lengths = np.fromiter((len(m) for m in member_lists), dtype=np.int64, count=len(member_lists))
    offsets = np.zeros(len(member_lists), dtype=np.int64)
    np.cumsum(lengths[:-1], out=offsets[1:])
    flat = np.concatenate(member_lists).astype(np.int64)
    return (
        torch.from_numpy(flat).to(device, non_blocking=True),
        torch.from_numpy(offsets).to(device, non_blocking=True),
        torch.from_numpy(np.asarray(extra_rows, dtype=np.float32)).to(device, non_blocking=True),
    )


@torch.no_grad()
def predict(net, member_lists, extra_rows, device, max_indices=2_000_000, max_bags=65536):
    """Predicted scores for many positions, split into chunks to limit memory."""
    out = np.empty(len(member_lists), dtype=np.float32)
    start = 0
    while start < len(member_lists):
        end, total = start, 0
        while end < len(member_lists) and end - start < max_bags and total + len(member_lists[end]) <= max_indices:
            total += len(member_lists[end])
            end += 1
        end = max(end, start + 1)
        idx, off, ex = batch_tensors(member_lists[start:end], extra_rows[start:end], device)
        out[start:end] = net(idx, off, ex).float().cpu().numpy()
        start = end
    return out
