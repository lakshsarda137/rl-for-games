"""PUCT Monte-Carlo Tree Search — the policy-improvement engine.

Each simulation: SELECT down the tree by PUCT, EXPAND+EVALUATE a leaf with one
network call (no rollouts — the value head is the score), then BACK UP the value.
The output is the root visit distribution, which is a stronger policy than the
raw network because it integrates look-ahead.

Perspective (consistent with encode.py): the evaluator returns a value from the
side-to-move's point of view. On backup, a leaf value `v` is added to edge
(node, a) as `+v` when that node's mover is the leaf's mover and `-v` otherwise
(zero-sum). Players are compared explicitly, so PASSES are handled correctly:
a forced pass is just a node whose only legal action is PASS (index 64), and
playing it hands the turn to the opponent like any other move.
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))

from board_numpy import apply_move, is_terminal, winner
from encode import POLICY_SIZE, legal_action_mask

_NEG_INF = -1e30


class _Node:
    __slots__ = ("player", "P", "N", "W", "legal", "children", "terminal", "tvalue")

    def __init__(self, player):
        self.player = player
        self.P = np.zeros(POLICY_SIZE, dtype=np.float32)
        self.N = np.zeros(POLICY_SIZE, dtype=np.float32)
        self.W = np.zeros(POLICY_SIZE, dtype=np.float32)
        self.legal = np.zeros(POLICY_SIZE, dtype=bool)
        self.children = {}
        self.terminal = False
        self.tvalue = 0.0


def _terminal_value(board, player):
    """Game result from `player`'s perspective: +1 win / -1 loss / 0 draw."""
    w = winner(board)
    if w == 0:
        return 0.0
    return 1.0 if w == player else -1.0


class MCTS:
    def __init__(self, evaluator, c_puct=1.5, dirichlet_alpha=0.3,
                 dirichlet_eps=0.25, rng=None):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = np.random.default_rng() if rng is None else rng

    def run(self, board, player, sims, add_noise=True):
        """Return root visit counts (length-65) after `sims` simulations."""
        return self.run_root(board, player, sims, add_noise).N.copy()

    def run_root(self, board, player, sims, add_noise=True):
        """Like `run`, but return the root node (for visit counts + root value)."""
        root = _Node(player)
        self._expand(root, board)
        if add_noise and not root.terminal:
            self._add_dirichlet(root)
        for _ in range(sims):
            self._simulate(root, board)
        return root

    def _expand(self, node, board):
        """Evaluate a leaf: set priors + legal mask (or terminal value)."""
        if is_terminal(board):
            node.terminal = True
            node.tvalue = _terminal_value(board, node.player)
            return node.tvalue
        priors, value = self.evaluator(board, node.player)
        node.P = priors.astype(np.float32)
        node.legal = legal_action_mask(board, node.player) > 0
        return value

    def _add_dirichlet(self, root):
        idx = np.where(root.legal)[0]
        if len(idx) == 0:
            return
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(idx)).astype(np.float32)
        root.P[idx] = (1 - self.dirichlet_eps) * root.P[idx] + self.dirichlet_eps * noise

    def _select(self, node):
        """PUCT: Q + c_puct * P * sqrt(sum N) / (1 + N), over legal actions."""
        sum_n = node.N.sum()
        if sum_n == 0:  # fresh node — follow the priors on the first pick
            scores = np.where(node.legal, node.P, _NEG_INF)
            return int(scores.argmax())
        q = np.zeros(POLICY_SIZE, dtype=np.float32)
        visited = node.N > 0
        q[visited] = node.W[visited] / node.N[visited]
        u = self.c_puct * node.P * math.sqrt(sum_n) / (1.0 + node.N)
        scores = q + u
        scores[~node.legal] = _NEG_INF
        return int(scores.argmax())

    def _simulate(self, root, root_board):
        node, board, path = root, root_board, []
        while True:
            if node.terminal:
                value, leaf_player = node.tvalue, node.player
                break
            action = self._select(node)
            path.append((node, action))
            board = apply_move(board, node.player, action)  # returns a fresh board
            child = node.children.get(action)
            if child is None:
                child = _Node(-node.player)  # a move or a pass both flip the mover
                node.children[action] = child
                value, leaf_player = self._expand(child, board), child.player
                break
            node = child
        for n, a in path:
            n.N[a] += 1.0
            n.W[a] += value if n.player == leaf_player else -value


def visit_policy(counts, temperature):
    """Turn root visit counts into a policy π over 65 actions.

    temperature=0 -> greedy (all mass on the most-visited action); otherwise
    π ∝ counts^(1/temperature), normalised over actions that were visited.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.sum() == 0:
        raise ValueError("no visits — cannot form a policy")
    if temperature <= 1e-8:
        pi = np.zeros_like(counts)
        pi[int(counts.argmax())] = 1.0
        return pi.astype(np.float32)
    scaled = counts ** (1.0 / temperature)
    return (scaled / scaled.sum()).astype(np.float32)


def root_value(root):
    """Search value estimate at the root, from the root mover's perspective."""
    total = root.N.sum()
    return float(root.W.sum() / total) if total > 0 else 0.0
