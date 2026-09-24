"""The network: one model, three answers about a position.

Input is what the player to move can see (native/rummy_native.cpp fill_obs):
10 card planes of 52 and 6 numbers. The 52 cards are laid out as a 4 x 13 grid
(suits x ranks), so a small convolutional net can spot runs along a row and
sets down a column. The ace is copied to a 14th column as well, so A-2-3 and
Q-K-A both look like neighbours.

Outputs:
  * policy: 54 scores, one per action (52 throw that card, 52 draw pile,
    53 discard pile). Illegal actions are masked out before the softmax.
  * value: how the hand will end for the player to move, in [-1, 1]
    (points won or lost / 130, the most a hand can cost).
  * hand guess: for each card, how likely it is in the opponent's hand. The
    search uses it to imagine the hidden cards; training checks it against
    the opponent's real hand.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))

from native import rummy_native as rn

N_PLANES, N_SCALARS, N_ACTIONS = rn.N_PLANES, rn.N_SCALARS, rn.N_ACTIONS


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(x + self.bn2(self.conv2(y)))


def card_grid(planes):
    """[B, C, 52] -> [B, C, 4, 14]: suits x ranks, with the ace also in column 13."""
    grid = planes.view(planes.shape[0], planes.shape[1], 4, 13)
    return torch.cat([grid, grid[..., :1]], dim=-1)


def fold_ace(grid):
    """[B, 4, 14] -> [B, 52]: add the high-ace column back onto the ace."""
    folded = grid[..., :13].clone()
    folded[..., 0] = folded[..., 0] + grid[..., 13]
    return folded.flatten(1)


class RummyNet(nn.Module):
    def __init__(self, blocks=6, channels=64):
        super().__init__()
        self.blocks_n, self.channels = blocks, channels
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES + N_SCALARS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.throw_head = nn.Conv2d(channels, 1, 1)     # one score per card
        self.hand_head = nn.Conv2d(channels, 1, 1)      # one guess per card
        self.draw_head = nn.Linear(channels, 2)         # draw pile, discard pile
        self.value_head = nn.Sequential(
            nn.Linear(channels, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Tanh())

    def forward(self, planes, scalars):
        """planes [B, 10, 52], scalars [B, 6] ->
        (policy logits [B, 54], value [B], hand logits [B, 52])."""
        x = card_grid(planes)
        s = scalars[:, :, None, None].expand(-1, -1, 4, 14)
        h = self.blocks(self.stem(torch.cat([x, s], dim=1)))
        pooled = h.mean(dim=(2, 3))
        throw = fold_ace(self.throw_head(h).squeeze(1))
        policy = torch.cat([throw, self.draw_head(pooled)], dim=1)
        value = self.value_head(pooled).squeeze(1)
        hand = fold_ace(self.hand_head(h).squeeze(1))
        return policy, value, hand


def masked_log_softmax(logits, legal):
    return F.log_softmax(logits.masked_fill(~legal, torch.finfo(logits.dtype).min), dim=-1)


class Evaluator:
    """The network as the C++ search calls it: NumPy in, NumPy out.

    use_hand_guess=False replaces the hand guess with "every card equally
    likely", for the comparison run that turns hand guessing off.
    """

    def __init__(self, net, device="cpu", use_hand_guess=True):
        self.net = net.to(device).eval()
        self.device = device
        self.use_hand_guess = use_hand_guess

    @torch.no_grad()
    def __call__(self, planes, scalars, legal):
        p = torch.from_numpy(np.ascontiguousarray(planes)).to(self.device)
        s = torch.from_numpy(np.ascontiguousarray(scalars)).to(self.device)
        m = torch.from_numpy(np.ascontiguousarray(legal)).to(self.device)
        logits, value, hand = self.net(p, s)
        priors = torch.exp(masked_log_softmax(logits.float(), m))
        guess = torch.sigmoid(hand.float()) if self.use_hand_guess else torch.ones_like(hand, dtype=torch.float32)
        return (priors.cpu().numpy().astype(np.float32),
                value.float().cpu().numpy().astype(np.float32),
                guess.cpu().numpy().astype(np.float32))


def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"   # Apple's MPS is slower than CPU for nets this small
