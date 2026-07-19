"""Rolling replay buffer of training examples.

Each example is (planes, pi, mask, z):
  planes float32 [NUM_PLANES,8,8] — the encoded position (side-to-move POV)
  pi     float32 [65]            — the MCTS search policy (training target)
  mask   bool    [65]            — legal actions (for masking the policy loss)
  z      float32 scalar          — game result from that state's mover's POV

A simple FIFO: the newest ~BUFFER_SIZE examples, so training always sees fresh
self-play. `sample` returns stacked torch tensors ready for a training step.
"""

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, planes, pi, mask, z):
        self.buffer.append((
            np.asarray(planes, dtype=np.float32),
            np.asarray(pi, dtype=np.float32),
            np.asarray(mask, dtype=bool),
            np.float32(z)))

    def extend(self, examples):
        for ex in examples:
            self.add(*ex)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size, device="cpu"):
        """A random minibatch as (planes, pi, mask, z) torch tensors."""
        n = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, n)
        planes, pi, mask, z = zip(*batch)
        return (
            torch.from_numpy(np.stack(planes)).to(device),
            torch.from_numpy(np.stack(pi)).to(device),
            torch.from_numpy(np.stack(mask)).to(device),
            torch.from_numpy(np.array(z, dtype=np.float32)).to(device))
