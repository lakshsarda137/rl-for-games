"""All AlphaZero hyperparameters in one place (plan §11).

`Config()` holds the full-scale starting values meant for a GPU (Kaggle).
`Config.tiny()` returns a shrunken config for local CPU smoke-runs and tests —
the pipeline must run end to end in seconds, so everything is scaled down.
"""

from dataclasses import dataclass, field, replace


@dataclass
class Config:
    # Network
    num_blocks: int = 5
    channels: int = 64

    # MCTS
    sims_selfplay: int = 160
    sims_eval: int = 400
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    # Self-play
    games_per_iter: int = 200
    temp_moves: int = 16          # plies played at tau=1 before switching to greedy
    augment: bool = True          # expand each example into its 8 dihedral symmetries
    selfplay_concurrency: int = 128  # games played in parallel == self-play net batch
                                     # size. Batched inference feeds the GPU: a batch
                                     # of N boards costs ~the same as one. Capped at
                                     # games_per_iter; raise both to fill a bigger GPU.
    selfplay_arrayops: bool = True   # self-play via batched engine + batched MCTS
                                     # (engine/board_batched.py + az/mcts_batched.py): the whole
                                     # game batch is searched with array ops, killing the
                                     # per-game Python overhead that is ~87% of self-play
                                     # time. The big throughput lever. False = coroutine pool.
    selfplay_workers: int = 1        # self-play worker PROCESSES. >1 spreads self-play over that
                                     # many CPU cores — but ONLY helps a CPU-ONLY run. On a GPU the
                                     # workers all share one device and contend (net eval serialises
                                     # across processes), so >1 is SLOWER there (2.1→1.3 g/s on a T4).
                                     # Keep 1 on GPU; raise only when device='cpu'.

    # Replay buffer
    buffer_size: int = 200_000

    # Training
    batch_size: int = 512
    steps_per_iter: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Evaluation / ladder
    eval_games: int = 100
    eval_depths: tuple = (2, 4, 6, 8)
    promote_threshold: float = 0.55

    # Loop / infra
    iterations: int = 40
    device: str = "cpu"
    seed: int = 0

    @classmethod
    def tiny(cls, **overrides):
        """A tiny config that runs the whole loop in seconds on CPU (tests/smoke).

        Defaults to the coroutine pool path (`selfplay_arrayops=False`) because the
        per-seed parity tests exercise it; the array-ops path has its own tests.
        """
        base = cls(
            num_blocks=2, channels=16,
            sims_selfplay=16, sims_eval=24,
            games_per_iter=4, temp_moves=6,
            buffer_size=5_000,
            batch_size=32, steps_per_iter=10,
            eval_games=4, eval_depths=(1, 2),
            iterations=2,
            selfplay_arrayops=False,
        )
        return replace(base, **overrides) if overrides else base

    @classmethod
    def kaggle(cls, **overrides):
        """First real GPU run: the 5x64 net, sized to climb the ladder within one
        free Kaggle session.

        Self-play runs the array-ops path (`selfplay_arrayops=True`): the whole
        96-game batch is searched with batched engine + batched MCTS, so the
        per-game Python overhead that dominated (batching the network alone left it
        CPU-bound at ~0.4 games/sec) is gone. This is the real throughput lever and
        should push max_depth_beaten toward depth-4; raise games_per_iter /
        sims_selfplay for a longer, stronger run.
        """
        base = cls(
            num_blocks=5, channels=64,          # the real network
            sims_selfplay=96, sims_eval=128,
            games_per_iter=96, temp_moves=12,
            selfplay_arrayops=True,             # array-ops self-play (batched engine + MCTS)
            selfplay_workers=1,                 # 1 on GPU: workers>1 all share ONE GPU and CONTEND
                                                # (measured 2.1 g/s at w=1 vs 1.3 at w=4 on a T4).
            buffer_size=100_000,
            batch_size=256, steps_per_iter=250,
            eval_games=20, eval_depths=(1, 2, 4),
            iterations=30,
            device="cuda",
        )
        return replace(base, **overrides) if overrides else base
