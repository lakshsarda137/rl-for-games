"""Training step and loss for the policy+value net.

Loss per example (plan §8.5):
    L = (z - v)^2  -  π · log p  +  c·||θ||^2
      = value MSE   +  policy cross-entropy  +  L2 (via optimizer weight_decay)

The policy cross-entropy is masked to legal actions: illegal logits are set to
-inf before log-softmax, and the target π is already 0 on illegal actions.

Learning-rate decay lives here too (`lr_at_iteration`): a cosine decay from
`cfg.lr` down to `cfg.lr_final` over `cfg.lr_horizon` global iterations. It is a
PURE FUNCTION of the iteration counter, NOT a stateful torch scheduler — the run
resumes across Kaggle sessions, and a scheduler would reset to `cfg.lr` on every
resume unless its state were persisted. Deriving the LR from the iteration (which
the checkpoint already stores) needs no extra state, so the decay is seamless
across resumes. Note `weight_decay` (L2) is a SEPARATE knob, not LR decay.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from network import masked_log_softmax


def make_optimizer(net, cfg):
    return torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def lr_at_iteration(cfg, iteration):
    """Cosine-decayed learning rate for a GLOBAL iteration (1-based).

    Decays smoothly from `cfg.lr` at iteration 1 down to `cfg.lr_final`, reaching
    the floor at iteration `cfg.lr_horizon` and holding there for the rest of an
    open-ended run. Being a pure function of `iteration` (stored in every
    checkpoint) is what makes it resume-safe: no scheduler state to save/restore,
    so the decay continues exactly where it left off after a Kaggle resume.

    Decay is DISABLED (constant `cfg.lr`) when `lr_final >= lr` or `lr_horizon <=
    1`, so the old constant-LR behaviour is one config away.
    """
    lr0 = cfg.lr
    lr1 = getattr(cfg, "lr_final", lr0)
    horizon = getattr(cfg, "lr_horizon", 0)
    if lr1 >= lr0 or horizon <= 1:
        return lr0
    # progress in [0, 1]: iteration 1 -> 0 (start), iteration `horizon` -> 1 (floor);
    # clamp past the horizon so an open-ended run simply holds lr1.
    t = min(max(iteration - 1, 0), horizon - 1) / (horizon - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))   # 1.0 at t=0 -> 0.0 at t=1
    return lr1 + (lr0 - lr1) * cosine


def set_lr(optimizer, lr):
    """Set `lr` on every optimizer param group; returns `lr` (handy for logging)."""
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


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
