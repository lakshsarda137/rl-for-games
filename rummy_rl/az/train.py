"""Training: a replay buffer of self-play positions and the loss.

Loss for each position (all three parts train together, every step):
    move choice   cross-entropy between the network's policy and the search's visits
    result        squared error between the value and how the hand really ended
    hand guess    how wrong the guess of the opponent's hand was (binary
                  cross-entropy), counted only on cards the player couldn't see
plus L2 weight decay through the optimizer.

The learning rate follows a cosine curve from cfg.lr down to cfg.lr_final,
computed from the iteration number alone, so resuming a run continues it.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from network import masked_log_softmax

UNSEEN_PLANE = 8   # plane 8 of the observation: cards the player can't locate


class ReplayBuffer:
    """The most recent `capacity` positions, stored compactly (0/1 planes as bytes)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.next = 0
        self.data = {
            "planes": np.zeros((capacity, 10, 52), np.uint8),
            "scalars": np.zeros((capacity, 6), np.float32),
            "legal": np.zeros((capacity, 54), bool),
            "pi": np.zeros((capacity, 54), np.float32),
            "z": np.zeros(capacity, np.float32),
            "hand": np.zeros((capacity, 52), np.uint8),
        }

    def __len__(self):
        return self.size

    def add(self, examples):
        n = len(examples["z"])
        for start in range(0, n, self.capacity):
            chunk = {k: v[start:start + self.capacity] for k, v in examples.items()}
            m = len(chunk["z"])
            idx = (self.next + np.arange(m)) % self.capacity
            for k, v in chunk.items():
                self.data[k][idx] = v
            self.next = (self.next + m) % self.capacity
            self.size = min(self.size + m, self.capacity)

    def sample(self, batch_size, rng, device):
        idx = rng.integers(0, self.size, size=batch_size)
        to = lambda a, dtype: torch.from_numpy(a[idx]).to(device=device, dtype=dtype)
        return (to(self.data["planes"], torch.float32), to(self.data["scalars"], torch.float32),
                to(self.data["legal"], torch.bool), to(self.data["pi"], torch.float32),
                to(self.data["z"], torch.float32), to(self.data["hand"], torch.float32))


def loss_batch(net, batch, hand_weight):
    planes, scalars, legal, pi, z, hand = batch
    logits, value, hand_logits = net(planes, scalars)
    policy_loss = -(pi * masked_log_softmax(logits, legal)).sum(1).mean()
    value_loss = F.mse_loss(value, z)
    unseen = planes[:, UNSEEN_PLANE]
    bce = F.binary_cross_entropy_with_logits(hand_logits, hand, reduction="none")
    hand_loss = (bce * unseen).sum() / unseen.sum().clamp(min=1)
    total = policy_loss + value_loss + hand_weight * hand_loss
    return total, {"policy": policy_loss.item(), "value": value_loss.item(), "hand": hand_loss.item()}


def lr_at_iteration(cfg, iteration):
    if cfg.lr_final >= cfg.lr or cfg.lr_horizon <= 1:
        return cfg.lr
    t = min(max(iteration - 1, 0), cfg.lr_horizon - 1) / (cfg.lr_horizon - 1)
    return cfg.lr_final + (cfg.lr - cfg.lr_final) * 0.5 * (1 + math.cos(math.pi * t))


def train_steps(net, buffer, optimizer, cfg, rng):
    net.train()
    totals = {"policy": 0.0, "value": 0.0, "hand": 0.0}
    steps = 0
    for _ in range(cfg.steps_per_iter):
        if len(buffer) < cfg.batch_size:
            break
        total, parts = loss_batch(net, buffer.sample(cfg.batch_size, rng, cfg.device), cfg.hand_weight)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        for k in totals:
            totals[k] += parts[k]
        steps += 1
    return {k: v / max(steps, 1) for k, v in totals.items()} | {"steps": steps}
