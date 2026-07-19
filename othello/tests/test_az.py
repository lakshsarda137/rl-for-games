"""AlphaZero pipeline tests. Run: python tests/test_az.py

FAST tier exercises every component cheaply, including the key acceptance check
(overfit-tiny: the net drives loss to ~0 on a handful of fixed positions, which
proves the training path works). SLOW tier runs the whole train loop end to end.
"""

import os
import sys
import tempfile

_HERE = os.path.dirname(__file__)
for _p in ("engine", "opponents", "az", "run"):
    sys.path.insert(0, os.path.join(_HERE, "..", _p))
sys.path.insert(0, _HERE)

import numpy as np
import torch

from board_numpy import BLACK, apply_move, initial_board, legal_moves
from encode import encode, legal_action_mask
from config import Config
from network import Evaluator, OthelloNet
from mcts import MCTS, visit_policy
from replay_buffer import ReplayBuffer
from selfplay import play_game
from train import loss_batch, make_optimizer, train_steps

from harness import check, run


def _tiny_net(seed=0):
    torch.manual_seed(seed)
    return OthelloNet(num_blocks=2, channels=16)


# --- components (fast) -------------------------------------------------------
def test_network_and_evaluator():
    net = _tiny_net()
    logits, v = net(torch.zeros(3, 3, 8, 8))
    check("policy head is 65-wide", logits.shape == (3, 65))
    check("value head is per-example scalar", v.shape == (3,))
    check("value is in [-1, 1]", bool((v >= -1).all() and (v <= 1).all()))

    board = initial_board()
    priors, val = Evaluator(net)(board, BLACK)
    legal = legal_moves(board, BLACK)
    illegal = [i for i in range(64) if i not in legal]
    check("priors sum to 1 over legal actions", abs(float(priors.sum()) - 1.0) < 1e-5)
    check("no prior mass on illegal actions", float(priors[illegal].sum()) < 1e-6)
    check("value is a finite scalar", np.isfinite(val))


def test_mcts_basic():
    net = _tiny_net()
    mcts = MCTS(Evaluator(net), rng=np.random.default_rng(0))
    board = initial_board()
    counts = mcts.run(board, BLACK, sims=40, add_noise=True)
    legal = set(legal_moves(board, BLACK))
    check("visit counts sum to sims", int(counts.sum()) == 40)
    check("visits only on legal actions",
          all(counts[a] == 0 for a in range(65) if a not in legal))
    check("greedy visit_policy is one-hot on a legal move",
          int(visit_policy(counts, 0.0).argmax()) in legal)
    check("tau=1 policy sums to 1", abs(float(visit_policy(counts, 1.0).sum()) - 1.0) < 1e-5)


def test_replay_buffer():
    buf = ReplayBuffer(100)
    for _ in range(20):
        buf.add(np.zeros((3, 8, 8), np.float32), np.zeros(65, np.float32),
                np.ones(65, bool), 1.0)
    planes, pi, mask, z = buf.sample(8)
    check("buffer sample planes shape", tuple(planes.shape) == (8, 3, 8, 8))
    check("buffer sample policy shape", tuple(pi.shape) == (8, 65))
    check("buffer respects capacity", ReplayBuffer(5).buffer.maxlen == 5 and len(buf) == 20)


def test_selfplay_produces_valid_examples():
    net = _tiny_net()
    cfg = Config.tiny()
    ex, rec = play_game(Evaluator(net), cfg, np.random.default_rng(1), make_record=True)
    check("self-play produced examples", len(ex) > 0)
    check("augmentation gives 8x per position", len(ex) == rec["plies"] * 8)
    planes, pi, mask, z = ex[0]
    check("example planes shape", planes.shape == (3, 8, 8))
    check("example policy sums to 1", abs(float(pi.sum()) - 1.0) < 1e-4)
    check("z is a game result", z in (-1.0, 0.0, 1.0))
    check("record has per-move mcts_value + top_policy",
          "mcts_value" in rec["moves"][0] and "top_policy" in rec["moves"][0])


def test_overfit_tiny():
    """The net memorises a handful of fixed positions -> loss ~0 (training works)."""
    net = _tiny_net()
    board, p = initial_board(), BLACK
    planes, pis, masks, zs = [], [], [], []
    for k in range(8):
        legal = legal_moves(board, p)
        planes.append(encode(board, p))
        masks.append(legal_action_mask(board, p) > 0)
        pi = np.zeros(65, np.float32); pi[legal[0]] = 1.0
        pis.append(pi); zs.append(np.float32(1.0 if k % 2 else -1.0))
        board = apply_move(board, p, legal[0]); p = -p
    batch = (torch.tensor(np.stack(planes)), torch.tensor(np.stack(pis)),
             torch.tensor(np.stack(masks)), torch.tensor(np.array(zs)))
    opt = make_optimizer(net, Config.tiny())
    net.train()
    first = None
    for step in range(400):
        opt.zero_grad(); total, _ = loss_batch(net, batch); total.backward(); opt.step()
        if step == 0:
            first = float(total)
    check(f"overfit-tiny loss collapses ({first:.2f} -> {float(total):.4f})", float(total) < 0.05)


# --- whole loop (slow) -------------------------------------------------------
def test_train_loop_end_to_end():
    from train_loop import train
    out = tempfile.mkdtemp(prefix="az_test_")
    cfg = Config.tiny(iterations=2)
    net, buf, hist = train(cfg, out_dir=out, eval_every=2, log=False, verbose=False)
    check("loop produced one row per iteration", len(hist) == 2)
    check("losses are finite", all(np.isfinite(r["loss"]["total"]) for r in hist))
    check("buffer grew across iterations", hist[1]["buffer"] > hist[0]["buffer"])
    check("ladder eval ran (max_depth_beaten present)", "max_depth_beaten" in hist[1])
    check("checkpoints written",
          len([f for f in os.listdir(os.path.join(out, "checkpoints")) if f.endswith(".pt")]) == 2)
    check("game records written",
          len([f for f in os.listdir(os.path.join(out, "game_records")) if f.endswith(".json")]) > 0)


FAST = [test_network_and_evaluator, test_mcts_basic, test_replay_buffer,
        test_selfplay_produces_valid_examples, test_overfit_tiny]
SLOW = [test_train_loop_end_to_end]

if __name__ == "__main__":
    run(FAST, SLOW, "az")
