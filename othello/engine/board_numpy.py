"""Reference Othello (Reversi) engine — the correctness oracle for the whole project.

Everything else consumes this module: the minimax opponent, the AlphaZero encoder
and MCTS, the web app, and the PyCUDA parity test all trust it to define the rules.
So it favours obvious correctness over speed.

Conventions locked here (read `encode.py`'s docstring for the *perspective* convention,
which is the other half of the contract):

  * Board = numpy int8 array, shape (8, 8), indexed `board[row, col]`.
    Cell values use ABSOLUTE colours, not "me / opponent":
        EMPTY = 0,  BLACK = +1,  WHITE = -1.
    Storing absolute colours keeps this engine simple; canonicalising to the
    side-to-move's point of view is `encode.py`'s job, done in exactly one place.

  * A `player` argument is always BLACK (+1) or WHITE (-1). Because colours are
    +1 / -1, "the opponent" is just `-player`.

  * A *square move* is an index 0..63 with `move = row * 8 + col` (row-major).
    The explicit PASS action (index 64) lives in `encode.py` with the rest of the
    65-wide action space; this engine handles passing through `must_pass` /
    `apply_move(..., PASS)` but never lists a pass among `legal_moves`.

  * `apply_move` never mutates its input — it returns a fresh board — so search
    code can share positions freely.

Terminal rule (easy to get wrong, easy to drop when porting to CUDA later):
a game ends when *neither* side has a legal move. That is USUALLY a full board,
but not always (a side can be wiped out, or both sides stuck early). Never
shortcut this to "board is full".
"""

import numpy as np

# --- board geometry & colours ------------------------------------------------
BOARD_N = 8
EMPTY = 0
BLACK = 1
WHITE = -1

# The explicit pass action index. Defined here as well as in encode.py so the
# engine can accept it in apply_move; encode.py owns the full 65-action space.
PASS = BOARD_N * BOARD_N  # 64

# The 8 straight-line directions (dr, dc).
_DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


# --- square <-> (row, col) indexing (locked before the NN planes exist) ------
def rc_to_move(row, col):
    """(row, col) -> square index 0..63."""
    return row * BOARD_N + col


def move_to_rc(move):
    """Square index 0..63 -> (row, col)."""
    return divmod(move, BOARD_N)


# --- core rules --------------------------------------------------------------
def initial_board():
    """Standard Othello opening: the four centre discs, Black to move."""
    board = np.zeros((BOARD_N, BOARD_N), dtype=np.int8)
    board[3, 3] = WHITE
    board[3, 4] = BLACK
    board[4, 3] = BLACK
    board[4, 4] = WHITE
    return board


def _flips_for(board, player, row, col):
    """Discs flipped if `player` plays at (row, col); [] if the move is illegal.

    A square is a legal move iff it is empty and, in at least one direction, a
    contiguous run of opponent discs is capped by one of the player's own discs.
    """
    if board[row, col] != EMPTY:
        return []
    opp = -player
    flips = []
    for dr, dc in _DIRECTIONS:
        r, c = row + dr, col + dc
        run = []
        while 0 <= r < BOARD_N and 0 <= c < BOARD_N and board[r, c] == opp:
            run.append((r, c))
            r += dr
            c += dc
        # A run counts only if it is non-empty AND terminated by our own disc
        # (not by the board edge or an empty square).
        if run and 0 <= r < BOARD_N and 0 <= c < BOARD_N and board[r, c] == player:
            flips.extend(run)
    return flips


def legal_moves(board, player):
    """Sorted list of square indices 0..63 where `player` may place a disc.

    An empty list means the player has no placement and must pass (see
    `must_pass`). Passing is never returned here — it is a separate action.
    """
    moves = []
    for row in range(BOARD_N):
        for col in range(BOARD_N):
            if board[row, col] == EMPTY and _flips_for(board, player, row, col):
                moves.append(rc_to_move(row, col))
    return moves


def legal_move_mask(board, player):
    """Boolean (8, 8) array marking squares where `player` may place a disc."""
    mask = np.zeros((BOARD_N, BOARD_N), dtype=bool)
    for move in legal_moves(board, player):
        r, c = move_to_rc(move)
        mask[r, c] = True
    return mask


def apply_move(board, player, move):
    """Return a NEW board after `player` plays `move` (a square 0..63, or PASS).

    PASS is only valid when the player genuinely has no legal move; applying a
    pass while placements exist is a bug, so it is asserted against.
    """
    if move == PASS:
        assert not legal_moves(board, player), "cannot pass when a legal move exists"
        return board.copy()

    row, col = move_to_rc(move)
    flips = _flips_for(board, player, row, col)
    assert flips, f"illegal move {move} for player {player}"
    new_board = board.copy()
    new_board[row, col] = player
    for r, c in flips:
        new_board[r, c] = player
    return new_board


def must_pass(board, player):
    """True if `player` has no legal placement (they must pass this turn)."""
    return not legal_moves(board, player)


def is_terminal(board):
    """True iff NEITHER colour has a legal move (see the module terminal-rule note)."""
    return must_pass(board, BLACK) and must_pass(board, WHITE)


# --- results / scoring -------------------------------------------------------
def count_discs(board):
    """(black_count, white_count)."""
    return int(np.sum(board == BLACK)), int(np.sum(board == WHITE))


def disc_diff(board, player):
    """Signed disc margin from `player`'s perspective: (player discs) - (opp discs).

    Used by minimax leaf evaluation (real win/loss margins should dominate
    heuristic noise) and by eval reporting. Positive = `player` is ahead.
    """
    black, white = count_discs(board)
    signed = black - white  # from Black's perspective
    return signed if player == BLACK else -signed


def winner(board):
    """Final result by disc count: BLACK (+1), WHITE (-1), or 0 for a draw.

    Only meaningful once `is_terminal(board)` is True.
    """
    black, white = count_discs(board)
    if black > white:
        return BLACK
    if white > black:
        return WHITE
    return 0


# --- human-readable rendering (leaned on by every failing test below) --------
_GLYPH = {EMPTY: ".", BLACK: "X", WHITE: "O"}


def render(board, player=None):
    """ASCII board. If `player` is given, its legal moves are marked with '*'.

    X = Black, O = White, . = empty, * = a legal move for `player`.
    """
    marks = legal_move_mask(board, player) if player is not None else None
    lines = ["  " + " ".join(str(c) for c in range(BOARD_N))]
    for r in range(BOARD_N):
        cells = []
        for c in range(BOARD_N):
            if marks is not None and marks[r, c]:
                cells.append("*")
            else:
                cells.append(_GLYPH[int(board[r, c])])
        lines.append(f"{r} " + " ".join(cells))
    if player is not None:
        black, white = count_discs(board)
        who = "Black(X)" if player == BLACK else "White(O)"
        lines.append(f"  X:{black} O:{white}  to move: {who}")
    return "\n".join(lines)
