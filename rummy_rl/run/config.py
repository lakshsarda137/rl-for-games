"""All training settings in one place.

Config()          a full run on a GPU (Kaggle).
Config.tiny()     a few-minute check on a laptop CPU that the whole system runs
                  and learns something, before spending GPU hours.
"""

from dataclasses import dataclass, replace


@dataclass
class Config:
    # Network
    blocks: int = 6
    channels: int = 64

    # Search
    worlds: int = 8               # imagined versions of the hidden cards per decision
    sims: int = 32                # look-ahead steps in each world
    c_puct: float = 1.5           # how much the search trusts the network's move scores vs. trying new moves
    dir_alpha: float = 0.5        # random noise on the first move scores in self-play, so it explores
    dir_eps: float = 0.25
    temp_moves: int = 10          # decisions per hand picked at random in proportion to visits

    # Self-play
    selfplay_games: int = 256     # hands played side by side (also the network's batch size)
    hands_per_iter: int = 512
    turn_cap: int = 200           # T9.1

    # Training
    buffer_size: int = 400_000
    batch_size: int = 512
    steps_per_iter: int = 400
    lr: float = 1e-3
    lr_final: float = 1e-4
    lr_horizon: int = 150         # iteration where the learning rate reaches lr_final
    weight_decay: float = 1e-4
    hand_weight: float = 1.0      # weight of the hand-guess loss; 0 turns hand guessing off

    # Checking strength against the greedy bot
    eval_every: int = 5           # iterations between checks (0 = never)
    eval_pairs: int = 200         # duplicate pairs, so 2x this many hands
    eval_sims: int = 32
    eval_worlds: int = 8

    device: str = "auto"
    use_hand_guess: bool = True   # False = search imagines hidden cards uniformly (comparison run)

    @classmethod
    def tiny(cls):
        return replace(cls(), blocks=2, channels=16, worlds=2, sims=6, selfplay_games=32,
                       hands_per_iter=32, buffer_size=50_000, batch_size=128, steps_per_iter=40,
                       lr_horizon=20, eval_every=2, eval_pairs=40, eval_sims=0)

    @classmethod
    def kaggle(cls):
        return replace(cls(), selfplay_games=512, hands_per_iter=1024, steps_per_iter=600)
