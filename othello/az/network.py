"""Policy + value network — a small ResNet, the heart of the AlphaZero agent.

Input: encoded planes from the side-to-move's perspective ([NUM_PLANES, 8, 8]).
Two heads:
  * policy — 65 logits (64 squares + pass). Illegal moves are masked to -inf
    *outside* the net (at MCTS/training time) before the softmax, so the net
    itself always emits raw logits.
  * value  — a scalar in [-1, 1] via tanh: the expected result for the side to
    move (+1 = side-to-move wins). This matches the perspective convention fixed
    in encode.py; MCTS negates it per ply.

Start small (NUM_BLOCKS=5, CHANNELS=64). Othello is tiny — scale up only if the
loss curves say the net is underfitting.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))

from encode import NUM_PLANES, POLICY_SIZE, encode, legal_action_mask


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class OthelloNet(nn.Module):
    def __init__(self, num_blocks=5, channels=64):
        super().__init__()
        self.num_blocks, self.channels = num_blocks, channels
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # Policy head: 1x1 conv -> flatten -> linear to 65 logits.
        self.p_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(inplace=True))
        self.p_fc = nn.Linear(2 * 8 * 8, POLICY_SIZE)

        # Value head: 1x1 conv -> flatten -> linear -> scalar -> tanh.
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(inplace=True))
        self.v_fc = nn.Sequential(
            nn.Linear(8 * 8, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        """x: [B, NUM_PLANES, 8, 8] -> (policy_logits [B, 65], value [B])."""
        h = self.blocks(self.stem(x))
        policy = self.p_fc(self.p_conv(h).flatten(1))
        value = self.v_fc(self.v_conv(h).flatten(1)).squeeze(1)
        return policy, value


def masked_log_softmax(logits, legal_mask):
    """log-softmax over legal actions only (illegal -> 0 probability)."""
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~legal_mask, neg_inf)
    return F.log_softmax(logits, dim=-1)


class Evaluator:
    """Wraps a net for MCTS: one board -> (priors over 65 actions, value).

    Priors are a proper distribution over *legal* actions (0 on illegal ones);
    value is from the side-to-move's perspective. Single-position eval — fine at
    tiny scale; batch later for throughput.
    """

    def __init__(self, net, device="cpu"):
        self.net = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, board, player):
        planes = encode(board, player)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        mask = torch.from_numpy(legal_action_mask(board, player)).to(self.device).bool()
        logp = masked_log_softmax(logits[0], mask)
        priors = torch.exp(logp).cpu().numpy().astype(np.float32)
        return priors, float(value.item())
