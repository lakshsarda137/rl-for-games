"""PUCT Monte-Carlo Tree Search — the policy-improvement engine.

Each simulation: SELECT down the tree by PUCT, EXPAND+EVALUATE a leaf with one
network call (no rollouts — the value head is the score), then BACK UP the value.
The output is the root visit distribution, which is a stronger policy than the
raw network because it integrates look-ahead.

The search is written as a COROUTINE (`run_root_gen`): instead of calling the
network itself, it *yields* a leaf's `(board, player)` and waits to be handed
back `(priors, value)` via `.send()`. This lets a driver run many independent
searches at once and evaluate ALL their pending leaves in one batched network
call — the fix for the GPU-starvation bottleneck (a GPU evals 256 boards ≈ as
fast as 1). The serial `run`/`run_root` keep the old one-call-per-leaf interface
by driving the same coroutine with the single-board `evaluator`, so there is a
single search implementation. Within any one game the simulations stay strictly
sequential, so batching across games leaves each game's search identical (bar
float32 rounding in the batched matmul, ~1e-7, which cannot flip a move once
root Dirichlet noise has broken any ties).

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
    def __init__(self, evaluator=None, c_puct=1.5, dirichlet_alpha=0.3,
                 dirichlet_eps=0.25, rng=None):
        # `evaluator` may be None: the coroutine path (batched self-play) never
        # calls it — the driver answers leaf requests instead. It's only needed
        # for the serial `run`/`run_root` convenience wrappers below.
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = np.random.default_rng() if rng is None else rng

    def run(self, board, player, sims, add_noise=True):
        """Return root visit counts (length-65) after `sims` simulations."""
        return self.run_root(board, player, sims, add_noise).N.copy()

    def run_root(self, board, player, sims, add_noise=True):
        """Serial search: drive the coroutine, answering each leaf with one net
        call via `self.evaluator`. Returns the root node."""
        return _drive_serial(self.run_root_gen(board, player, sims, add_noise),
                             self.evaluator)

    def run_root_gen(self, board, player, sims, add_noise=True):
        """Coroutine search. Yields `(board, player)` leaf-eval requests and
        expects `(priors, value)` back via `.send()`; returns the root node on
        completion. Drive it serially with `_drive_serial`, or batch many of
        these together (az/selfplay.py) to feed a GPU."""
        root = _Node(player)
        yield from self._expand_gen(root, board)
        if add_noise and not root.terminal:
            self._add_dirichlet(root)
        for _ in range(sims):
            yield from self._simulate_gen(root, board)
        return root

    def _expand_gen(self, node, board):
        """Evaluate a leaf: set priors + legal mask (or terminal value).

        Yields the leaf's `(board, player)` for the driver to evaluate; a
        terminal leaf needs no net call and returns its exact result instead."""
        if is_terminal(board):
            node.terminal = True
            node.tvalue = _terminal_value(board, node.player)
            return node.tvalue
        priors, value = yield (board, node.player)
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

    def _simulate_gen(self, root, root_board):
        """One simulation as a coroutine: SELECT to a leaf, yield it for
        EVALUATE (unless terminal), then BACK UP. Yields at most once."""
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
                value = yield from self._expand_gen(child, board)
                leaf_player = child.player
                break
            node = child
        self._backup(path, value, leaf_player)

    @staticmethod
    def _backup(path, value, leaf_player):
        """Add the leaf value along the visited path, sign per node's mover."""
        for n, a in path:
            n.N[a] += 1.0
            n.W[a] += value if n.player == leaf_player else -value


def _drive_serial(gen, evaluator):
    """Run a search coroutine to completion, answering each yielded leaf with a
    single-board `evaluator(board, player)` call. Returns the coroutine's value
    (the root node). Used by the non-batched paths (eval, single-game play)."""
    try:
        request = gen.send(None)
    except StopIteration as done:
        return done.value
    while True:
        board, player = request
        result = evaluator(board, player)
        try:
            request = gen.send(result)
        except StopIteration as done:
            return done.value


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
