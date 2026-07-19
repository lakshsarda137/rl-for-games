"""8-fold dihedral (D4) symmetry for boards and 65-length policies.

Othello is invariant under the 8 symmetries of the square (4 rotations x 2
reflections). We exploit this for 8x data augmentation of training examples, and
optionally for averaging network evaluations. The board and its policy MUST be
transformed *consistently* — a disc and the move-probability on that square move
together — so both live here and are tested jointly (apply-then-invert = id).

Policy layout (from encode.py): indices 0..63 are squares, index 64 is PASS.
PASS is a FIXED POINT under every symmetry — the transforms permute the 64
square indices but leave index 64 alone. This is the easy off-by-one: never
reshape a 65-vector into a grid.
"""

import numpy as np

from encode import PASS, POLICY_SIZE

NUM_SYMMETRIES = 8

# Base square transforms on an (8, 8) array. The reflection variants are the
# rotations of a left-right flip, giving the full D4 group. Copies keep results
# contiguous (rot90 / fliplr return views with awkward strides).
_TRANSFORMS = (
    lambda b: np.ascontiguousarray(b),
    lambda b: np.ascontiguousarray(np.rot90(b, 1)),
    lambda b: np.ascontiguousarray(np.rot90(b, 2)),
    lambda b: np.ascontiguousarray(np.rot90(b, 3)),
    lambda b: np.ascontiguousarray(np.fliplr(b)),
    lambda b: np.ascontiguousarray(np.rot90(np.fliplr(b), 1)),
    lambda b: np.ascontiguousarray(np.rot90(np.fliplr(b), 2)),
    lambda b: np.ascontiguousarray(np.rot90(np.fliplr(b), 3)),
)

# For each symmetry i, `_SQUARE_PERM[i]` is a length-64 array where entry (r*8+c)
# after transform came from original square `_SQUARE_PERM[i][r*8+c]`. Built by
# transforming an index-labelled board, so the policy permutation is guaranteed
# to match the board transform exactly.
_LABELLED = np.arange(64, dtype=np.int64).reshape(8, 8)
_SQUARE_PERM = tuple(t(_LABELLED).reshape(-1) for t in _TRANSFORMS)

# Inverse symmetry index, computed (not hand-reasoned) so it cannot drift: j
# undoes i iff applying j after i returns the identity labelling.
def _compute_inverses():
    inverses = [None] * NUM_SYMMETRIES
    for i in range(NUM_SYMMETRIES):
        after_i = _TRANSFORMS[i](_LABELLED)
        for j in range(NUM_SYMMETRIES):
            if np.array_equal(_TRANSFORMS[j](after_i), _LABELLED):
                inverses[i] = j
                break
        assert inverses[i] is not None, f"no inverse found for symmetry {i}"
    return tuple(inverses)


_INVERSE = _compute_inverses()


def transform_board(board, i):
    """Apply symmetry `i` (0..7) to an (8, 8) board."""
    return _TRANSFORMS[i](board)


def inverse_index(i):
    """The symmetry that undoes symmetry `i`."""
    return _INVERSE[i]


def transform_policy(policy, i):
    """Apply symmetry `i` to a length-65 policy; PASS (index 64) is unchanged.

    Entry at square s after the transform takes the value from the original
    square that mapped onto s, matching `transform_board`.
    """
    policy = np.asarray(policy)
    assert policy.shape[-1] == POLICY_SIZE, f"expected length {POLICY_SIZE} policy"
    out = np.empty_like(policy, dtype=policy.dtype)
    out[..., :PASS] = policy[..., _SQUARE_PERM[i]]
    out[..., PASS] = policy[..., PASS]  # PASS is a fixed point
    return out


def all_board_symmetries(board):
    """The 8 symmetric boards (in symmetry-index order)."""
    return [transform_board(board, i) for i in range(NUM_SYMMETRIES)]
