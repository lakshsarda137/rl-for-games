"""External RL opponent: alpha-zero-general's pretrained Othello net as a player.

This lets you benchmark your bot against *another* AlphaZero-family agent (an
RL peer), not just the search-based Edax. It vendors the exact PyTorch network
architecture from github.com/suragnair/alpha-zero-general (Othello, 8x8) so that
repo's pretrained `8x8_100checkpoints_best.pth.tar` state-dict loads directly, and
wraps it as an evaluator in OUR convention so it plugs into OUR MCTS. Running the
foreign net through our own search means a FAIR comparison: same sims, same PUCT —
only the learned network differs.

Encoding bridge (their net <-> our engine):
  * Input: their net takes a single 8x8 channel of the CANONICAL board (+1 = the
    side-to-move's discs, -1 = opponent, 0 = empty). We build that as board*player
    (our BLACK=+1/WHITE=-1 absolute board times the mover), no 3-plane encode().
  * Output: log_softmax over 65 actions (0..63 = row*8+col squares, 64 = pass) and
    a tanh value from the side-to-move's POV — the SAME action indexing and value
    perspective as ours, so priors/value map across with no permutation.
  * We mask to our legal actions and renormalise, exactly like our own evaluator.

Get the weights (one 64MB file; not committed — data/ is gitignored):
    curl -L -o data/external_models/azg_8x8.pth.tar \\
      https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/othello/pytorch/8x8_100checkpoints_best.pth.tar
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))
sys.path.insert(0, _HERE)

from encode import legal_action_mask
from mcts import MCTS

BOARD_N = 8
ACTION_SIZE = 65        # 64 squares + pass
NUM_CHANNELS = 512      # alpha-zero-general's default for the 8x8 Othello net
DROPOUT = 0.3


class OthelloNNet(nn.Module):
    """Byte-compatible copy of alpha-zero-general's Othello PyTorch net (4 conv +
    2 FC). Layer names/shapes match that repo so its checkpoint loads as-is.

    Input:  [B,8,8] canonical board (+1 = side-to-move's discs).
    Output: (log_softmax policy [B,65], tanh value [B,1]).
    """

    def __init__(self, board_n=BOARD_N, action_size=ACTION_SIZE,
                 num_channels=NUM_CHANNELS, dropout=DROPOUT):
        super().__init__()
        self.board_x = self.board_y = board_n
        self.action_size = action_size
        self.num_channels = num_channels
        self.dropout = dropout

        self.conv1 = nn.Conv2d(1, num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        flat = num_channels * (board_n - 4) * (board_n - 4)
        self.fc1 = nn.Linear(flat, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.num_channels * (self.board_x - 4) * (self.board_y - 4))
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)
        pi = self.fc3(s)
        v = self.fc4(s)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


def load_azg_net(path, device="cpu"):
    """Load an alpha-zero-general Othello checkpoint into a fresh net (eval mode).

    Uses `weights_only=True` first (safe: no arbitrary pickle code runs when
    loading a foreign file), falling back only if the payload isn't plain tensors.
    The repo saves `{'state_dict': ...}`; a bare state-dict is also accepted.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"external model not found: {path}. Download it with:\n"
            "  curl -L -o data/external_models/azg_8x8.pth.tar "
            "https://github.com/suragnair/alpha-zero-general/raw/master/"
            "pretrained_models/othello/pytorch/8x8_100checkpoints_best.pth.tar")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device)   # local, user-supplied file
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # Infer width + action size from the checkpoint so ANY alpha-zero-general Othello
    # net loads (their default is 512 channels / 65 actions), not just one hardcoded size.
    num_channels = int(state["conv1.weight"].shape[0])
    action_size = int(state["fc3.weight"].shape[0])
    net = OthelloNNet(num_channels=num_channels, action_size=action_size).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def azg_evaluator(net, device="cpu"):
    """Wrap the foreign net as OUR evaluator: `(board, player) -> (priors[65], value)`.

    priors is a proper distribution over OUR legal actions (0 on illegal), value is
    from the side-to-move's POV — the interface MCTS/az_player expect.
    """
    @torch.no_grad()
    def evaluate(board, player):
        canonical = np.asarray(board, dtype=np.float32) * float(player)  # +1 = mover's discs
        x = torch.from_numpy(canonical).to(device).view(1, BOARD_N, BOARD_N)
        log_pi, v = net(x)
        pi = torch.exp(log_pi)[0].detach().cpu().numpy().astype(np.float32)
        legal = legal_action_mask(board, player) > 0
        pi = np.where(legal, pi, 0.0).astype(np.float32)
        total = pi.sum()
        if total > 0:
            pi /= total
        else:                                   # net gave ~0 mass to legal moves
            pi = legal.astype(np.float32)
            pi /= max(pi.sum(), 1.0)
        return pi, float(np.asarray(v).reshape(-1)[0])

    return evaluate


def azg_player(path, sims, device="cpu", c_puct=1.5, rng=None, net=None):
    """A move function `(board, player) -> move` for the external net, driven by
    OUR MCTS at `sims` simulations (greedy on visits, no root noise) — so it's a
    fair same-search opponent. Pass a preloaded `net` to skip re-reading the file."""
    if net is None:
        net = load_azg_net(path, device)
    evaluator = azg_evaluator(net, device)
    mcts = MCTS(evaluator, c_puct=c_puct, rng=rng or np.random.default_rng())

    def move_fn(board, player):
        counts = mcts.run(board, player, sims, add_noise=False)
        return int(counts.argmax())            # tau=0: most-visited (always legal)

    return move_fn
