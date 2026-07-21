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

import contextlib
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


def inference_autocast(device, enabled):
    """FP16 autocast for net INFERENCE on CUDA (T4 tensor cores); no-op otherwise.

    FP16 only speeds up conv/linear matmuls on a CUDA GPU, so it's enabled ONLY
    there; on CPU (tests, local runs) this returns a null context and everything
    stays FP32 — keeping CPU play bit-exact with the parity oracles. Under autocast
    the heavy conv/linear ops run in half precision while BatchNorm and softmax stay
    FP32 (autocast's own policy); callers additionally cast logits/values back to
    float32 right after the forward, so only the forward matmuls are ever FP16.

    Trade-off: FP16 rounding (~1e-3) is far coarser than FP32's ~1e-7, so it is NOT
    bit-identical to the FP32 net — that's why it is opt-in and off by default.
    """
    if enabled and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class Evaluator:
    """Wraps a net for MCTS: boards -> (priors over 65 actions, value).

    Priors are a proper distribution over *legal* actions (0 on illegal ones);
    value is from the side-to-move's perspective. `evaluate_batch` runs many
    positions through the net in ONE forward pass — a GPU evaluates 256 boards
    at ~the cost of one, so batching many concurrent self-play games' pending
    MCTS leaves here is what keeps the GPU fed (see az/selfplay.py). `__call__`
    is the single-position convenience used by the serial (eval) path.

    Because the net is in eval mode, BatchNorm uses fixed running statistics, so
    a board's output is independent of what else shares its batch — batched eval
    matches one-at-a-time eval to within float32 rounding (~1e-7, from the matmul
    reduction order). That is far below self-play's own noise, so play strength is
    unaffected; MCTS tie-breaking is protected by root Dirichlet noise anyway.
    """

    def __init__(self, net, device="cpu", fp16=False):
        self.net = net.to(device).eval()
        self.device = device
        self.fp16 = fp16   # FP16 forward on CUDA (no-op on CPU); see inference_autocast

    @torch.no_grad()
    def evaluate_batch(self, boards, players):
        """Evaluate many (board, player) at once -> list of (priors, value).

        One net forward pass over the whole batch; results are returned in the
        input order, each priors a length-65 float32 distribution over legal
        actions and value a Python float from that board's mover's perspective.
        """
        planes = np.stack([encode(b, p) for b, p in zip(boards, players)])
        x = torch.from_numpy(planes).to(self.device)
        with inference_autocast(self.device, self.fp16):
            logits, values = self.net(x)
        logits, values = logits.float(), values.float()  # back to FP32 for the softmax
        masks = np.stack([legal_action_mask(b, p) for b, p in zip(boards, players)])
        mask = torch.from_numpy(masks).to(self.device).bool()
        priors = torch.exp(masked_log_softmax(logits, mask)).cpu().numpy().astype(np.float32)
        values = values.cpu().numpy().astype(np.float32)
        return [(priors[i], float(values[i])) for i in range(len(boards))]

    def __call__(self, board, player):
        (priors, value), = self.evaluate_batch([board], [player])
        return priors, value
