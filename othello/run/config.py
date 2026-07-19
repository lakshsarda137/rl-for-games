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
        """A tiny config that runs the whole loop in seconds on CPU (tests/smoke)."""
        base = cls(
            num_blocks=2, channels=16,
            sims_selfplay=16, sims_eval=24,
            games_per_iter=4, temp_moves=6,
            buffer_size=5_000,
            batch_size=32, steps_per_iter=10,
            eval_games=4, eval_depths=(1, 2),
            iterations=2,
        )
        return replace(base, **overrides) if overrides else base

    @classmethod
    def kaggle(cls, **overrides):
        """First real GPU run: the 5x64 net, sized to climb the ladder within one
        free Kaggle session.

        Batched self-play inference is now in place (games_per_iter games play
        concurrently and share one net call per step, see az/selfplay.py), so the
        GPU is fed instead of idling. `selfplay_concurrency == games_per_iter`
        here means the whole iteration's games run as one wave (a ~96-board eval
        batch). This should push max_depth_beaten past depth-2, toward depth-4;
        raise games_per_iter / sims_selfplay for a longer, stronger run.
        """
        base = cls(
            num_blocks=5, channels=64,          # the real network
            sims_selfplay=96, sims_eval=128,
            games_per_iter=96, temp_moves=12,
            selfplay_concurrency=96,            # one wave: ~96 boards per net call
            buffer_size=100_000,
            batch_size=256, steps_per_iter=250,
            eval_games=20, eval_depths=(1, 2, 4),
            iterations=30,
            device="cuda",
        )
        return replace(base, **overrides) if overrides else base
