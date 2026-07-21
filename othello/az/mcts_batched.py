"""Batched PUCT MCTS — B independent game trees searched in lockstep with array
ops. This is the throughput fix: the serial `mcts.py` runs SELECT/EXPAND/BACKUP as
per-game Python objects (a `_Node` per node, dict child lookups, tiny-array math),
and that per-item Python cost — not the network — is ~87% of self-play time. Here
the B trees live in flat `[B, max_nodes, 65]` arrays and every simulation step is
one set of NumPy ops over the whole batch, so that cost is paid once per step
instead of B times.

Semantics are identical to `mcts.py` (verified in `tests/test_batched.py`):
  * SELECT is the same PUCT rule (`Q + c_puct·P·√ΣN/(1+N)`, priors on a fresh node),
    same argmax tie-break.
  * A leaf is EXPANDED into one new node and EVALUATED — terminal nodes take their
    exact game result, others one batched network call (all games' pending leaves
    in a single forward pass).
  * BACKUP adds the leaf value along the visited path, `+v` where the node's mover
    matches the leaf's mover else `-v` (zero-sum, pass-safe) — never assuming
    strict colour alternation.

`run_batched` returns each game's root visit counts `[B, 65]` and root value `[B]`.
The evaluator is injected as `evaluate(boards, players) -> (priors[k,65], values[k])`
so tests can feed a bit-exact single-board evaluator (making the batched search
identical to serial) while production feeds the true batched network call.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))

import board_batched as bb
from encode import POLICY_SIZE

_NEG_INF = -1e30


def _terminal_values(boards, players):
    """Result from each game's `player` POV: +1 win / -1 loss / 0 draw ([k])."""
    w = bb.winner(boards)
    return np.where(w == 0, 0.0, np.where(w == players, 1.0, -1.0)).astype(np.float32)


def _select_actions(P, N, W, legal, c_puct):
    """PUCT action per game, given the current node's `[B,65]` stats."""
    sum_n = N.sum(1)
    sqrt_sum = np.sqrt(sum_n)[:, None]
    q = np.divide(W, N, out=np.zeros_like(W), where=N > 0)
    u = c_puct * P * sqrt_sum / (1.0 + N)
    scores = q + u
    scores = np.where((sum_n == 0)[:, None], P, scores)   # fresh node → follow priors
    scores = np.where(legal, scores, _NEG_INF)
    return scores.argmax(1)


def _add_dirichlet(P, legal, node, cfg, rng):
    """Mix root Dirichlet noise over legal actions (per game). Terminal roots have
    no legal actions, so their priors are left untouched."""
    lmask = legal[:, node]
    alpha, eps = cfg.dirichlet_alpha, cfg.dirichlet_eps
    g = (rng.gamma(alpha, 1.0, size=lmask.shape).astype(np.float32) * lmask)
    s = g.sum(1, keepdims=True)
    noise = np.divide(g, s, out=np.zeros_like(g), where=s > 0)
    P[:, node] = np.where(lmask, (1.0 - eps) * P[:, node] + eps * noise, P[:, node])


def _eval_into(gidx, nidx, boards, players, evaluate, P, legal,
               node_terminal, node_tvalue):
    """Fill node (gidx, nidx) for each game: terminal → exact value; else one
    batched network eval → priors + legal mask. Returns each game's leaf value."""
    term = bb.is_terminal(boards)
    values = np.zeros(len(gidx), np.float32)
    if term.any():
        tv = _terminal_values(boards[term], players[term])
        node_terminal[gidx[term], nidx[term]] = True
        node_tvalue[gidx[term], nidx[term]] = tv
        values[term] = tv
    live = ~term
    if live.any():
        priors, vals = evaluate(boards[live], players[live])
        P[gidx[live], nidx[live]] = priors
        legal[gidx[live], nidx[live]] = bb.legal_action_masks(boards[live],
                                                              players[live]) > 0
        values[live] = vals
    return values


def run_batched(boards, players, sims, evaluate, cfg, rng=None, add_noise=True):
    """Search B games for `sims` simulations each. Returns
    (visit_counts [B,65] float32, root_value [B] float32)."""
    boards = np.ascontiguousarray(boards, dtype=np.int8)
    players = np.asarray(players, dtype=np.int8)
    rng = rng or np.random.default_rng()
    B = boards.shape[0]
    A = POLICY_SIZE
    max_nodes = sims + 1
    ar = np.arange(B)

    P = np.zeros((B, max_nodes, A), np.float32)
    N = np.zeros((B, max_nodes, A), np.float32)
    W = np.zeros((B, max_nodes, A), np.float32)
    legal = np.zeros((B, max_nodes, A), bool)
    children = np.full((B, max_nodes, A), -1, np.int64)
    node_player = np.zeros((B, max_nodes), np.int8)
    node_board = np.zeros((B, max_nodes, bb.BOARD_N, bb.BOARD_N), np.int8)
    node_terminal = np.zeros((B, max_nodes), bool)
    node_tvalue = np.zeros((B, max_nodes), np.float32)
    n_nodes = np.ones(B, np.int64)          # root (node 0) is allocated

    # --- root ---
    node_board[:, 0] = boards
    node_player[:, 0] = players
    _eval_into(ar, np.zeros(B, np.int64), boards, players, evaluate,
               P, legal, node_terminal, node_tvalue)
    if add_noise:
        _add_dirichlet(P, legal, 0, cfg, rng)

    for _ in range(sims):
        # --- SELECT: descend every tree to its leaf, in lockstep ---
        cur = np.zeros(B, np.int64)
        plen = np.zeros(B, np.int64)
        path_nodes = np.full((B, max_nodes), -1, np.int64)
        path_actions = np.full((B, max_nodes), -1, np.int64)
        kind = np.zeros(B, np.int8)          # 1 = terminal leaf, 2 = expand leaf
        exp_parent = np.zeros(B, np.int64)
        exp_action = np.zeros(B, np.int64)
        leaf_value = np.zeros(B, np.float32)
        leaf_player = np.zeros(B, np.int8)
        active = np.ones(B, bool)

        while active.any():
            is_term = active & node_terminal[ar, cur]
            if is_term.any():
                kind[is_term] = 1
                leaf_value[is_term] = node_tvalue[ar[is_term], cur[is_term]]
                leaf_player[is_term] = node_player[ar[is_term], cur[is_term]]
                active = active & ~is_term
            if not active.any():
                break
            a = _select_actions(P[ar, cur], N[ar, cur], W[ar, cur],
                                legal[ar, cur], cfg.c_puct)
            aidx = np.where(active)[0]
            path_nodes[aidx, plen[aidx]] = cur[aidx]
            path_actions[aidx, plen[aidx]] = a[aidx]
            plen[aidx] += 1
            child = children[ar, cur, a]
            to_expand = active & (child < 0)
            if to_expand.any():
                kind[to_expand] = 2
                exp_parent[to_expand] = cur[to_expand]
                exp_action[to_expand] = a[to_expand]
            active = active & (child >= 0)
            cur = np.where(active, child, cur)

        # --- EXPAND + EVALUATE the new leaves (one batched net call) ---
        eb = np.where(kind == 2)[0]
        if len(eb):
            parent, act = exp_parent[eb], exp_action[eb]
            new_idx = n_nodes[eb].copy()
            children[eb, parent, act] = new_idx
            n_nodes[eb] += 1
            child_boards = bb.apply_moves(node_board[eb, parent],
                                          node_player[eb, parent], act)
            child_players = (-node_player[eb, parent]).astype(np.int8)
            node_board[eb, new_idx] = child_boards
            node_player[eb, new_idx] = child_players
            leaf_value[eb] = _eval_into(eb, new_idx, child_boards, child_players,
                                        evaluate, P, legal, node_terminal, node_tvalue)
            leaf_player[eb] = child_players

        # --- BACKUP along each path (sign per node's mover vs leaf's mover) ---
        for level in range(int(plen.max()) if B else 0):
            gi = np.where(level < plen)[0]
            if not len(gi):
                continue
            nodes, acts = path_nodes[gi, level], path_actions[gi, level]
            sign = np.where(node_player[gi, nodes] == leaf_player[gi],
                            1.0, -1.0).astype(np.float32)
            N[gi, nodes, acts] += 1.0
            W[gi, nodes, acts] += sign * leaf_value[gi]

    counts = N[:, 0, :].copy()
    total = counts.sum(1)
    root_value = np.divide(W[:, 0, :].sum(1), total,
                           out=np.zeros(B, np.float32), where=total > 0)
    return counts, root_value


def make_net_evaluator(net, device="cpu"):
    """Production evaluator: `(boards[k,8,8], players[k]) -> (priors[k,65], values[k])`
    in ONE forward pass, encoding via the batched engine."""
    import torch

    from network import masked_log_softmax
    from profiling import PROF

    @torch.no_grad()
    def evaluate(boards, players):
        # Profiling (opt-in, off by default): split the net FORWARD (GPU — the part a
        # C++ port can't speed up) from the encode/transfer prep. cuda.synchronize()
        # brackets the forward so its async kernels aren't misattributed downstream.
        if PROF.enabled:
            t0 = PROF.clock()
        planes = bb.encode_batch(boards, players)
        x = torch.from_numpy(planes).to(device)
        if PROF.enabled:
            PROF.sync(); tf = PROF.clock()
        logits, values = net(x)
        if PROF.enabled:
            PROF.sync()
            PROF.add("net_fwd", PROF.clock() - tf)
            PROF.count("fwd_calls"); PROF.count("positions", len(boards))
        mask = torch.from_numpy(bb.legal_action_masks(boards, players)).to(device).bool()
        priors = torch.exp(masked_log_softmax(logits, mask)).cpu().numpy().astype(np.float32)
        out = priors, values.cpu().numpy().astype(np.float32)
        if PROF.enabled:
            PROF.add("net_eval_total", PROF.clock() - t0)
        return out

    return evaluate
