"""Training step and loss for the policy+value net.

Loss per example (plan §8.5):
    L = (z - v)^2  -  π · log p  +  c·||θ||^2
      = value MSE   +  policy cross-entropy  +  L2 (via optimizer weight_decay)

The policy cross-entropy is masked to legal actions: illegal logits are set to
-inf before log-softmax, and the target π is already 0 on illegal actions.
"""

import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from network import masked_log_softmax


def make_optimizer(net, cfg):
    return torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def loss_batch(net, batch):
    """Return (total_loss, {policy, value}) for one (planes, pi, mask, z) batch."""
    planes, pi, mask, z = batch
    logits, v = net(planes)

    logp = masked_log_softmax(logits, mask)
    policy_loss = -(pi * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(v, z)
    total = policy_loss + value_loss
    return total, {"policy": float(policy_loss.item()), "value": float(value_loss.item())}


def train_steps(net, buffer, optimizer, cfg):
    """Run cfg.steps_per_iter minibatch updates; return averaged loss metrics."""
    net.train()
    agg = {"policy": 0.0, "value": 0.0, "total": 0.0}
    steps = 0
    for _ in range(cfg.steps_per_iter):
        if len(buffer) == 0:
            break
        batch = buffer.sample(cfg.batch_size, device=cfg.device)
        optimizer.zero_grad()
        total, parts = loss_batch(net, batch)
        total.backward()
        optimizer.step()
        agg["policy"] += parts["policy"]
        agg["value"] += parts["value"]
        agg["total"] += float(total.item())
        steps += 1
    if steps:
        for k in agg:
            agg[k] /= steps
    agg["steps"] = steps
    return agg
