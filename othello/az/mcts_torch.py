"""Torch port of the batched PUCT MCTS — the same B-trees-in-lockstep search as
`mcts_batched`, but every array is a **torch tensor** so the whole search (not
just the network) runs wherever the tensors live: CPU on a Mac, CUDA on Kaggle.

Why this exists (othello/CLAUDE.md next-steps item 3): `mcts_batched` is NumPy =
CPU. Batching removed the per-game Python overhead, but the tree-search + rules
math still runs on one CPU core, which is the throughput wall once the network is
on the GPU. This module is the op-for-op re-expression of that search in torch,
so the search itself can use the idle GPU. It is device-agnostic torch (NOT
PyCUDA): build + correctness-test on the Mac with CPU tensors, prove speed on a
GPU. The port is welded to running FAR bigger game batches (`_play_batch_torch`
in az/selfplay.py) — the per-step op-launch cost is fixed regardless of batch, so
you want each step doing many more games to amortise it on the GPU.

Semantics are identical to `mcts_batched` / `mcts.py` (parity-tested in
`tests/test_batched.py`):
  * SELECT is the same PUCT rule (`Q + c_puct·P·√ΣN/(1+N)`, priors on a fresh
    node) with **first-max** tie-breaking (`_argmax_first`) so it matches NumPy's
    `argmax` exactly even on the (rare) exact tie.
  * A leaf is EXPANDED into one new node and EVALUATED — terminal nodes take their
    exact game result, others one batched network call.
  * BACKUP adds the leaf value along the visited path, `+v` where the node's mover
    matches the leaf's mover else `-v` (zero-sum, pass-safe).

`run_torch` returns each game's root visit counts `[B, 65]` and root value `[B]`
as torch tensors on the input device. The evaluator is injected as
`evaluate(boards, players) -> (priors[k,65], values[k])` (all torch, on-device),
so tests can feed a bit-exact single-board evaluator while production feeds the
true batched network call (`make_net_evaluator_torch`).
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))

import board_torch as bt
from encode import POLICY_SIZE

_NEG_INF = -1e30


def _argmax_first(scores):
    """Row-wise argmax that returns the FIRST maximal index on ties, matching
    `numpy.argmax` exactly (torch.argmax's tie-break is unspecified). `scores`
    is `[B, A]`; returns `[B]` long."""
    a = scores.shape[1]
    is_max = scores == scores.max(dim=1, keepdim=True).values
    idx = torch.where(is_max,
                      torch.arange(a, device=scores.device).expand_as(is_max),
                      torch.full_like(is_max, a, dtype=torch.long))
    return idx.min(dim=1).values


def _terminal_values(boards, players):
    """Result from each game's `player` POV: +1 win / -1 loss / 0 draw ([k] float32)."""
    w = bt.winner(boards).to(torch.int8)
    ones = torch.ones(w.shape, dtype=torch.float32, device=w.device)
    vals = torch.where(w == players.to(torch.int8), ones, -ones)
    return torch.where(w == 0, torch.zeros_like(vals), vals)


def _select_actions(P, N, W, legal, c_puct):
    """PUCT action per game, given the current node's `[B,65]` stats."""
    sum_n = N.sum(1)
    sqrt_sum = torch.sqrt(sum_n)[:, None]
    safe_n = torch.where(N > 0, N, torch.ones_like(N))      # avoid 0/0; masked below
    q = torch.where(N > 0, W / safe_n, torch.zeros_like(W))
    u = c_puct * P * sqrt_sum / (1.0 + N)
    scores = q + u
    scores = torch.where((sum_n == 0)[:, None], P, scores)  # fresh node → follow priors
    scores = torch.where(legal, scores, torch.full_like(scores, _NEG_INF))
    return _argmax_first(scores)


def _add_dirichlet(P, legal, node, cfg, rng):
    """Mix root Dirichlet noise over legal actions (per game). Uses the NumPy `rng`
    for the gamma draw so the noise is identical to `mcts_batched._add_dirichlet`
    (same seed → same values), keeping the torch path reproducible/comparable."""
    lmask = legal[:, node]                                  # [B, A] bool
    alpha, eps = cfg.dirichlet_alpha, cfg.dirichlet_eps
    g_np = rng.gamma(alpha, 1.0, size=tuple(lmask.shape)).astype(np.float32)
    g = torch.from_numpy(g_np).to(P.device) * lmask.to(torch.float32)
    s = g.sum(1, keepdim=True)
    noise = torch.where(s > 0, g / torch.where(s > 0, s, torch.ones_like(s)),
                        torch.zeros_like(g))
    P[:, node] = torch.where(lmask, (1.0 - eps) * P[:, node] + eps * noise, P[:, node])


def _eval_into(gidx, nidx, boards, players, evaluate, P, legal,
               node_terminal, node_tvalue):
    """Fill node (gidx, nidx) for each game: terminal → exact value; else one
    batched network eval → priors + legal mask. Returns each game's leaf value."""
    term = bt.is_terminal(boards)
    values = torch.zeros(len(gidx), dtype=torch.float32, device=boards.device)
    if bool(term.any()):
        tv = _terminal_values(boards[term], players[term])
        node_terminal[gidx[term], nidx[term]] = True
        node_tvalue[gidx[term], nidx[term]] = tv
        values[term] = tv
    live = ~term
    if bool(live.any()):
        priors, vals = evaluate(boards[live], players[live])
        P[gidx[live], nidx[live]] = priors.to(torch.float32)
        legal[gidx[live], nidx[live]] = bt.legal_action_masks(boards[live],
                                                              players[live]) > 0
        values[live] = vals.to(torch.float32)
    return values


def run_torch(boards, players, sims, evaluate, cfg, rng=None, add_noise=True):
    """Search B games for `sims` simulations each (torch). Returns
    (visit_counts [B,65] float32, root_value [B] float32) on the input device.

    `boards` is int8 `[B,8,8]`, `players` int8 `[B]`; both may be torch tensors
    (any device) or NumPy — they are moved to a single common device (the boards'
    device if a tensor, else CPU). `evaluate(boards, players)` must return torch
    `(priors[k,65], values[k])` on that device (see `make_net_evaluator_torch`)."""
    if not torch.is_tensor(boards):
        boards = torch.as_tensor(np.ascontiguousarray(boards, dtype=np.int8))
    boards = boards.to(torch.int8).contiguous()
    device = boards.device
    players = torch.as_tensor(players).to(device=device, dtype=torch.int8)
    rng = rng or np.random.default_rng()
    B = boards.shape[0]
    A = POLICY_SIZE
    max_nodes = sims + 1
    ar = torch.arange(B, device=device)

    P = torch.zeros((B, max_nodes, A), dtype=torch.float32, device=device)
    N = torch.zeros((B, max_nodes, A), dtype=torch.float32, device=device)
    W = torch.zeros((B, max_nodes, A), dtype=torch.float32, device=device)
    legal = torch.zeros((B, max_nodes, A), dtype=torch.bool, device=device)
    children = torch.full((B, max_nodes, A), -1, dtype=torch.long, device=device)
    node_player = torch.zeros((B, max_nodes), dtype=torch.int8, device=device)
    node_board = torch.zeros((B, max_nodes, bt.BOARD_N, bt.BOARD_N),
                             dtype=torch.int8, device=device)
    node_terminal = torch.zeros((B, max_nodes), dtype=torch.bool, device=device)
    node_tvalue = torch.zeros((B, max_nodes), dtype=torch.float32, device=device)
    n_nodes = torch.ones(B, dtype=torch.long, device=device)   # root (node 0) allocated

    # --- root ---
    node_board[:, 0] = boards
    node_player[:, 0] = players
    _eval_into(ar, torch.zeros(B, dtype=torch.long, device=device), boards, players,
               evaluate, P, legal, node_terminal, node_tvalue)
    if add_noise:
        _add_dirichlet(P, legal, 0, cfg, rng)

    for _ in range(sims):
        # --- SELECT: descend every tree to its leaf, in lockstep ---
        cur = torch.zeros(B, dtype=torch.long, device=device)
        plen = torch.zeros(B, dtype=torch.long, device=device)
        path_nodes = torch.full((B, max_nodes), -1, dtype=torch.long, device=device)
        path_actions = torch.full((B, max_nodes), -1, dtype=torch.long, device=device)
        kind = torch.zeros(B, dtype=torch.int8, device=device)   # 1=terminal, 2=expand
        exp_parent = torch.zeros(B, dtype=torch.long, device=device)
        exp_action = torch.zeros(B, dtype=torch.long, device=device)
        leaf_value = torch.zeros(B, dtype=torch.float32, device=device)
        leaf_player = torch.zeros(B, dtype=torch.int8, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)

        while bool(active.any()):
            is_term = active & node_terminal[ar, cur]
            if bool(is_term.any()):
                kind[is_term] = 1
                leaf_value[is_term] = node_tvalue[ar[is_term], cur[is_term]]
                leaf_player[is_term] = node_player[ar[is_term], cur[is_term]]
                active = active & ~is_term
            if not bool(active.any()):
                break
            a = _select_actions(P[ar, cur], N[ar, cur], W[ar, cur],
                                legal[ar, cur], cfg.c_puct)
            aidx = torch.nonzero(active, as_tuple=True)[0]
            path_nodes[aidx, plen[aidx]] = cur[aidx]
            path_actions[aidx, plen[aidx]] = a[aidx]
            plen[aidx] += 1
            child = children[ar, cur, a]
            to_expand = active & (child < 0)
            if bool(to_expand.any()):
                kind[to_expand] = 2
                exp_parent[to_expand] = cur[to_expand]
                exp_action[to_expand] = a[to_expand]
            active = active & (child >= 0)
            cur = torch.where(active, child, cur)

        # --- EXPAND + EVALUATE the new leaves (one batched net call) ---
        eb = torch.nonzero(kind == 2, as_tuple=True)[0]
        if eb.numel():
            parent, act = exp_parent[eb], exp_action[eb]
            new_idx = n_nodes[eb].clone()
            children[eb, parent, act] = new_idx
            n_nodes[eb] += 1
            child_boards = bt.apply_moves(node_board[eb, parent],
                                          node_player[eb, parent], act)
            child_players = (-node_player[eb, parent]).to(torch.int8)
            node_board[eb, new_idx] = child_boards
            node_player[eb, new_idx] = child_players
            leaf_value[eb] = _eval_into(eb, new_idx, child_boards, child_players,
                                        evaluate, P, legal, node_terminal, node_tvalue)
            leaf_player[eb] = child_players

        # --- BACKUP along each path (sign per node's mover vs leaf's mover) ---
        max_len = int(plen.max()) if B else 0
        for level in range(max_len):
            gi = torch.nonzero(level < plen, as_tuple=True)[0]
            if gi.numel() == 0:
                continue
            nodes = path_nodes[gi, level]
            acts = path_actions[gi, level]
            ones = torch.ones(gi.shape, dtype=torch.float32, device=device)
            sign = torch.where(node_player[gi, nodes] == leaf_player[gi], ones, -ones)
            N[gi, nodes, acts] += 1.0
            W[gi, nodes, acts] += sign * leaf_value[gi]

    counts = N[:, 0, :].clone()
    total = counts.sum(1)
    root_value = torch.where(total > 0, W[:, 0, :].sum(1) / torch.where(
        total > 0, total, torch.ones_like(total)), torch.zeros(B, dtype=torch.float32,
                                                               device=device))
    return counts, root_value


def make_net_evaluator_torch(net, device="cpu"):
    """Production evaluator for `run_torch`:
    `(boards[k,8,8] int8, players[k] int8) -> (priors[k,65], values[k])` in ONE
    forward pass, all torch and **kept on `device`** (no NumPy round-trip inside
    the search). Encodes via the torch batched engine."""
    from network import masked_log_softmax

    @torch.no_grad()
    def evaluate(boards, players):
        planes = bt.encode_batch(boards, players)
        logits, values = net(planes)
        mask = bt.legal_action_masks(boards, players).bool()
        priors = torch.exp(masked_log_softmax(logits, mask))
        return priors, values

    return evaluate
