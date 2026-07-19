"""Board <-> neural-net encoding, and THE canonical statement of perspective.

This module owns two conventions that many other components must agree on. If
they ever disagree, you get silent sign errors in value targets that are
miserable to debug three files away — so they are fixed and tested here, once.

================================  ACTION SPACE  ===============================
The policy vector has length POLICY_SIZE = 65:
    indices 0..63  -> board squares, `action = row * 8 + col` (row-major),
    index   64     -> PASS.
Squares reuse the engine's `rc_to_move` / `move_to_rc`. Only PASS is new here.

===============================  PERSPECTIVE  ================================
The engine stores ABSOLUTE colours (Black = +1, White = -1). The network, by
contrast, ALWAYS sees the position from the point of view of the side to move —
it never learns "am I Black or White", only "me vs opponent". `encode(board,
player)` is the single place that performs this canonicalisation:

    plane 0 = `player`'s own discs        (1.0 where board == player)
    plane 1 = the opponent's discs        (1.0 where board == -player)
    plane 2 = `player`'s legal-move mask  (1.0 on legal placements)

Consequences that the rest of the system MUST honour:
  * The value head outputs v in [-1, 1] = "how good is this position for the
    side to move", NOT "how good for Black". So during MCTS backup the value is
    negated once per ply (players alternate), and the self-play result `z` is
    stamped per state from THAT state's mover's perspective: +1 if the mover of
    that state went on to win the game, -1 if they lost, 0 for a draw.
  * Because the board is canonicalised, an explicit "side to move" input plane
    would be constant (all ones) and carry no information, so it is omitted.
    NUM_PLANES is the single source of truth for the network's input channels.

`decode(planes, player)` inverts `encode` back to an absolute-colour board, so
the round-trip is testable.
"""

import numpy as np

from board_numpy import (
    BOARD_N,
    EMPTY,
    legal_move_mask,
    legal_moves,
    move_to_rc,
    rc_to_move,
)

# --- action space ------------------------------------------------------------
PASS = BOARD_N * BOARD_N       # 64
POLICY_SIZE = PASS + 1         # 65

# --- network input planes (single source of truth for input channels) --------
NUM_PLANES = 3


def encode(board, player):
    """Absolute-colour board -> float32 planes [NUM_PLANES, 8, 8] in `player`'s POV.

    See the module docstring: plane 0 = own discs, plane 1 = opponent discs,
    plane 2 = own legal-move mask. Always from the side-to-move's perspective.
    """
    planes = np.zeros((NUM_PLANES, BOARD_N, BOARD_N), dtype=np.float32)
    planes[0] = (board == player)
    planes[1] = (board == -player)
    planes[2] = legal_move_mask(board, player)
    return planes


def decode(planes, player):
    """Inverse of `encode`: planes (POV of `player`) -> absolute-colour int8 board."""
    own = planes[0].astype(np.int8)
    opp = planes[1].astype(np.int8)
    return (own * player + opp * (-player)).astype(np.int8)


def legal_action_mask(board, player):
    """Float32 mask over the 65-wide action space (1.0 = legal), including PASS.

    PASS (index 64) is legal exactly when the player has no placement, matching
    the engine's passing rule. Used to mask illegal logits to -inf before the
    policy softmax.
    """
    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
    moves = legal_moves(board, player)
    if moves:
        for move in moves:
            mask[move] = 1.0
    else:
        mask[PASS] = 1.0
    return mask


def action_to_rc(action):
    """Action index -> (row, col). PASS raises (it has no square)."""
    if action == PASS:
        raise ValueError("PASS has no board square")
    return move_to_rc(action)


def rc_to_action(row, col):
    """(row, col) -> square action index 0..63."""
    return rc_to_move(row, col)
