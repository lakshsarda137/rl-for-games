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

from board_numpy import BLACK, apply_move, initial_board, is_terminal, legal_moves
from encode import encode, legal_action_mask
from config import Config
from network import Evaluator, OthelloNet
from mcts import MCTS, visit_policy
from replay_buffer import ReplayBuffer
from selfplay import (
    _chunk,
    _play_batch,
    _play_batch_parallel,
    _play_parallel,
    _play_pool,
    generate_games,
    play_game,
    shutdown_selfplay_workers,
)
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


def test_evaluate_batch_matches_single():
    """Batched inference matches one-board-at-a-time (eval-mode BatchNorm uses
    fixed stats, so a board's output doesn't depend on its batch) — up to float32
    last-bit rounding from the batched matmul's reduction order (~1e-7). A masking
    or reshape bug would show as a difference orders of magnitude larger."""
    net = _tiny_net()
    ev = Evaluator(net)
    board, p, boards, players = initial_board(), BLACK, [], []
    for _ in range(6):                       # a handful of real, distinct positions
        boards.append(board.copy()); players.append(p)
        legal = legal_moves(board, p)
        if not legal:
            break
        board = apply_move(board, p, legal[0]); p = -p
    batched = ev.evaluate_batch(boards, players)      # one forward pass, many boards
    max_dp = max_dv = 0.0
    for (pb, vb), b, pl in zip(batched, boards, players):
        ps, vs = ev(b, pl)                            # a size-1 batch (via __call__)
        max_dp = max(max_dp, float(np.abs(pb - ps).max()))
        max_dv = max(max_dv, abs(vb - vs))
    check(f"batched priors match single-board eval (max dp {max_dp:.1e})", max_dp < 1e-5)
    check(f"batched value matches single-board eval (max dv {max_dv:.1e})", max_dv < 1e-5)


def test_batched_selfplay_matches_serial():
    """The play-quality guarantee: a batched game matches the serial game with the
    same seed, example for example — batching changes speed, not strength. It comes
    out byte-identical because the only difference (float32 matmul rounding, ~1e-7)
    is far too small to flip a PUCT argmax once root Dirichlet noise breaks ties."""
    net = _tiny_net()
    ev = Evaluator(net)
    # Kept cheap for the FAST tier: few sims, 3 games, 2 in flight -> still exercises
    # multi-game batching AND the "start a new game when one finishes" refill path.
    cfg = Config.tiny(sims_selfplay=4, selfplay_concurrency=2)
    seeds = [11, 22, 33]

    serial = [play_game(ev, cfg, np.random.default_rng(s))[0] for s in seeds]
    pooled = _play_pool(ev, cfg, seeds)               # concurrent, one net call per step

    check("pool returns one result per game", len(pooled) == len(seeds))
    identical = True
    for (ex_b, _rec, _plies), ex_s in zip(pooled, serial):
        identical = identical and len(ex_b) == len(ex_s) and all(
            np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            and np.array_equal(a[2], b[2]) and a[3] == b[3]
            for a, b in zip(ex_b, ex_s))
    check("batched self-play examples identical to serial, game for game", identical)

    # And the public generator is deterministic given the master rng.
    ex1, _, st1 = generate_games(ev, cfg, np.random.default_rng(7), 2, make_records=True)
    ex2, _, st2 = generate_games(ev, cfg, np.random.default_rng(7), 2, make_records=True)
    check("generate_games is reproducible for a fixed seed",
          len(ex1) == len(ex2) and st1["num_examples"] == st2["num_examples"]
          and all(np.array_equal(a[0], b[0]) for a, b in zip(ex1, ex2)))


def test_arrayops_selfplay_matches_serial_greedy():
    """Array-ops self-play (batched engine + batched MCTS, vectorised over games)
    plays the SAME game as the trusted serial MCTS when both are greedy and
    noise-free — an end-to-end check of the whole batched loop (search, passes,
    terminal detection, history, z-stamping), not just a component."""
    from types import SimpleNamespace
    net = _tiny_net()
    ev = Evaluator(net)
    cfg = SimpleNamespace(c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25,
                          sims_selfplay=12, temp_moves=0, augment=False)

    board, player, ref = initial_board(), BLACK, []      # deterministic serial game
    m = MCTS(ev, c_puct=1.5)
    while not is_terminal(board):
        move = int(m.run(board, int(player), cfg.sims_selfplay, add_noise=False).argmax())
        ref.append(move)
        board = apply_move(board, int(player), move); player = -player

    # greedy + no noise -> all games identical; must match the serial line move-for-move
    pg = _play_batch(ev, cfg, np.random.default_rng(0), 3, make_records=True, add_noise=False)
    check("array-ops greedy self-play == serial greedy, move-for-move",
          all([e["move"] for e in pg[g][1]["moves"]] == ref for g in range(3)))

    cfg_aug = SimpleNamespace(c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25,
                              sims_selfplay=12, temp_moves=4, augment=True)
    ex, rec, plies = _play_batch(ev, cfg_aug, np.random.default_rng(2), 4)[0]
    planes, pi, mask, z = ex[0]
    check("array-ops augments x8, valid planes/pi/z",
          len(ex) == plies * 8 and planes.shape == (3, 8, 8)
          and abs(float(pi.sum()) - 1.0) < 1e-4 and z in (-1.0, 0.0, 1.0))


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


# --- multiprocess self-play (slow: spawns worker processes) ------------------
def test_parallel_selfplay_matches_inprocess():
    """Multiprocess self-play produces exactly the games the single-process pool
    would. Each worker runs `_play_pool` on a seed-slice, so the parallel result
    equals the in-process chunk pools concatenated, byte-for-byte — validating the
    plumbing (weight transfer, seed chunking, order) without depending on
    float-rounding tie-breaks. This is the CPU-bound throughput lever."""
    net = _tiny_net()
    ev = Evaluator(net)
    cfg = Config.tiny(sims_selfplay=6, selfplay_concurrency=3)
    seeds = [int(s) for s in np.random.default_rng(5).integers(1, 2 ** 62, size=5)]

    ref = []
    for c in _chunk(seeds, 2):                 # what each worker will run, in-process
        ref += _play_pool(ev, cfg, c)
    try:
        par = _play_parallel(ev, cfg, seeds, 2)   # spawns 2 worker processes
        same = len(par) == len(ref) and all(
            len(a[0]) == len(b[0]) and all(
                np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
                and np.array_equal(x[2], y[2]) and x[3] == y[3]
                for x, y in zip(a[0], b[0]))
            for a, b in zip(par, ref))
        check("parallel workers reproduce the in-process pools byte-for-byte", same)
    finally:
        shutdown_selfplay_workers()            # reap the spawned processes


def test_arrayops_parallel_matches_inprocess():
    """Array-ops across worker PROCESSES (the 'use every core' path) produces
    exactly the games the in-process array-ops would for the same per-worker seeds
    — validates the multi-core plumbing (weight transfer, game-count chunking,
    result collection). With the single-process greedy==serial check, this covers
    array-ops both on one core and across many."""
    from types import SimpleNamespace
    net = _tiny_net()
    ev = Evaluator(net)
    cfg = SimpleNamespace(c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25,
                          sims_selfplay=6, temp_moves=3, augment=False,
                          selfplay_arrayops=True, selfplay_workers=2)
    num_games, workers = 5, 2

    rng = np.random.default_rng(0)             # replicate the parallel seed derivation
    sizes = [len(c) for c in _chunk(list(range(num_games)), workers)]
    seeds = rng.integers(1, np.iinfo(np.int64).max, size=len(sizes))
    ref = []
    for w in range(len(sizes)):
        ref += _play_batch(ev, cfg, np.random.default_rng(int(seeds[w])), sizes[w])
    try:
        par = _play_batch_parallel(ev, cfg, np.random.default_rng(0), num_games, workers)
        same = len(par) == len(ref) and all(
            len(a[0]) == len(b[0]) and all(
                np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
                and np.array_equal(x[2], y[2]) and x[3] == y[3]
                for x, y in zip(a[0], b[0]))
            for a, b in zip(par, ref))
        check("array-ops workers == in-process chunks byte-for-byte", same)
    finally:
        shutdown_selfplay_workers()


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
        test_selfplay_produces_valid_examples, test_evaluate_batch_matches_single,
        test_batched_selfplay_matches_serial, test_arrayops_selfplay_matches_serial_greedy,
        test_overfit_tiny]
SLOW = [test_parallel_selfplay_matches_inprocess, test_arrayops_parallel_matches_inprocess,
        test_train_loop_end_to_end]

if __name__ == "__main__":
    run(FAST, SLOW, "az")
