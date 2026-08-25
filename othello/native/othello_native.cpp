/*
  othello_native — the C++ (pybind11) port of Othello move generation + PUCT tree
  search. This is a drop-in replacement for the two hot NumPy modules:

      engine/board_batched.py   ->  the bitboard engine below
      az/mcts_batched.py        ->  run_batched() below

  WHY: on a Kaggle T4 the measured self-play split was net_fwd 43% (GPU forward,
  untouchable from here), tree_ops 37% + net_prep 16% + per_move 1% (NumPy on the
  CPU). Those CPU buckets are not slow because the arithmetic is heavy — they are
  slow because they dispatch millions of NumPy calls on ~6k-element arrays, where
  most of each call is interpreter + dispatch overhead. In C++ the same work is a
  handful of scalar loops over a few thousand floats.

  WHAT STAYS IN PYTHON (deliberately):
    * The network forward pass. `run_batched` takes an `evaluate` CALLBACK and
      calls back into Python once per simulation step, exactly where the NumPy
      version made its batched net call. So no LibTorch, no CUDA in this build —
      this is a plain CPU extension that links nothing but pybind11.
    * The root Dirichlet noise. It is the only part of the search that consumes
      the NumPy RNG, and reproducing PCG64's gamma stream here bit-for-bit would
      be a permanent parity liability for no speed gain (it is O(B*65) ONCE per
      search, against the O(sims*depth*B*65) inner loop C++ actually takes over).
      Python computes the normalised noise vector and passes it in.

  BIT-EXACT PARITY WITH NUMPY is the contract (tests/test_native.py pins it, and
  tests/test_batched.py pins the result against the serial mcts.py oracle). The
  things that make it hold:
    * Every float is `float` (never double). c_puct / dirichlet_eps are cast to
      float exactly where NumPy casts them, and `1.0 - eps` is computed in double
      first because NumPy computes that Python-scalar expression in float64.
    * Operator order matches NumPy's left-to-right evaluation, op for op.
    * `N.sum()` is order-independent here: N only ever holds non-negative INTEGER
      float32 values (+= 1.0), and integer sums below 2^24 are exact in float32.
      So NumPy's pairwise summation and this naive loop agree bit-for-bit.
    * argmax keeps NumPy's FIRST-maximum tie-break (strict `>` scanning upward).
    * Built with -ffp-contract=off, so no FMA silently fuses a multiply+add into
      a differently-rounded result. See native/build.py.

  Board representation at the Python boundary stays int8 [B, 8, 8] absolute
  colours (BLACK=+1, WHITE=-1, EMPTY=0), so nothing else in the codebase changes;
  the uint64 bitboards are an internal detail, packed/unpacked at the edge.
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

// --- constants (mirrors of the Python ones) ---------------------------------

constexpr int BOARD_N = 8;
constexpr int NSQ = 64;
constexpr int PASS = 64;                 // encode.py action convention
constexpr int POLICY_SIZE = 65;
constexpr int MAX_RUN = 6;               // board_batched._MAX_RUN
constexpr float NEG_INF = -1e30f;        // mcts_batched._NEG_INF

// Bit i of a uint64 is square (row, col) with i = row*8 + col, so a board row is
// one byte and `col` is the low 3 bits. These masks drop the squares that would
// wrap around the edge when shifting sideways.
constexpr uint64_t NOT_COL0 = 0xFEFEFEFEFEFEFEFEULL;   // col != 0
constexpr uint64_t NOT_COL7 = 0x7F7F7F7F7F7F7F7FULL;   // col != 7

inline int popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int c = 0;
    while (x) { x &= x - 1; ++c; }
    return c;
#endif
}

// --- bitboard engine (the board_batched.py port) ----------------------------

/* `_shift(plane, dr, dc)` in board_batched.py is `out[r][c] = plane[r-dr][c-dc]`,
   i.e. slide the contents by (dr, dc) with no wraparound. In bit terms the
   destination index is `i + dr*8 + dc`. D indexes board_batched._DIRS. */
template <int D>
inline uint64_t shift_dir(uint64_t x) {
    if (D == 0) return (x & NOT_COL0) >> 9;   // (-1, -1)
    if (D == 1) return  x             >> 8;   // (-1,  0)
    if (D == 2) return (x & NOT_COL7) >> 7;   // (-1, +1)
    if (D == 3) return (x & NOT_COL0) >> 1;   // ( 0, -1)
    if (D == 4) return (x & NOT_COL7) << 1;   // ( 0, +1)
    if (D == 5) return (x & NOT_COL0) << 7;   // (+1, -1)
    if (D == 6) return  x             << 8;   // (+1,  0)
    return             (x & NOT_COL7) << 9;   // (+1, +1)
}

/* One direction's legal placements: walk a contiguous opponent run outward from
   our own discs; the empty square just past it is a capture. */
template <int D>
inline uint64_t moves_in_dir(uint64_t own, uint64_t opp, uint64_t empty) {
    uint64_t run = shift_dir<D>(own) & opp;
    for (int k = 0; k < MAX_RUN; ++k) run |= shift_dir<D>(run) & opp;
    return shift_dir<D>(run) & empty;
}

inline uint64_t legal_moves_bb(uint64_t own, uint64_t opp) {
    const uint64_t empty = ~(own | opp);
    return moves_in_dir<0>(own, opp, empty) | moves_in_dir<1>(own, opp, empty)
         | moves_in_dir<2>(own, opp, empty) | moves_in_dir<3>(own, opp, empty)
         | moves_in_dir<4>(own, opp, empty) | moves_in_dir<5>(own, opp, empty)
         | moves_in_dir<6>(own, opp, empty) | moves_in_dir<7>(own, opp, empty);
}

/* The opponent run captured in direction D by placing at `placed` — empty unless
   one of our own discs caps the far end. */
template <int D>
inline uint64_t flips_in_dir(uint64_t own, uint64_t opp, uint64_t placed) {
    uint64_t run = shift_dir<D>(placed) & opp;
    uint64_t frontier = run;
    for (int k = 0; k < MAX_RUN; ++k) {
        frontier = shift_dir<D>(frontier) & opp;
        run |= frontier;
    }
    return (shift_dir<D>(run) & own) ? run : 0ULL;
}

inline uint64_t flips_bb(uint64_t own, uint64_t opp, uint64_t placed) {
    return flips_in_dir<0>(own, opp, placed) | flips_in_dir<1>(own, opp, placed)
         | flips_in_dir<2>(own, opp, placed) | flips_in_dir<3>(own, opp, placed)
         | flips_in_dir<4>(own, opp, placed) | flips_in_dir<5>(own, opp, placed)
         | flips_in_dir<6>(own, opp, placed) | flips_in_dir<7>(own, opp, placed);
}

struct BB { uint64_t black, white; };

inline BB unpack(const int8_t* p) {
    BB b{0ULL, 0ULL};
    for (int i = 0; i < NSQ; ++i) {
        const int8_t v = p[i];
        if (v > 0)      b.black |= (1ULL << i);
        else if (v < 0) b.white |= (1ULL << i);
    }
    return b;
}

inline void pack(const BB& b, int8_t* p) {
    for (int i = 0; i < NSQ; ++i) {
        const uint64_t m = 1ULL << i;
        p[i] = (b.black & m) ? int8_t(1) : ((b.white & m) ? int8_t(-1) : int8_t(0));
    }
}

inline void own_opp(const BB& b, int8_t player, uint64_t& own, uint64_t& opp) {
    if (player > 0) { own = b.black; opp = b.white; }
    else            { own = b.white; opp = b.black; }
}

/* Terminal = NEITHER side can move (not "board full") — board_batched.is_terminal. */
inline bool terminal_bb(const BB& b) {
    return legal_moves_bb(b.black, b.white) == 0ULL
        && legal_moves_bb(b.white, b.black) == 0ULL;
}

inline int winner_bb(const BB& b) {
    const int nb = popcount64(b.black), nw = popcount64(b.white);
    return (nb > nw) ? 1 : ((nw > nb) ? -1 : 0);
}

/* board_batched.apply_moves for one game: every square in (flips | placed)
   becomes the mover's colour; PASS leaves the board untouched. */
inline BB apply_bb(const BB& b, int8_t player, int64_t action) {
    if (action == PASS) return b;
    if (action < 0 || action > PASS)
        throw std::invalid_argument("action out of range [0, 64]");
    uint64_t own, opp;
    own_opp(b, player, own, opp);
    const uint64_t placed = 1ULL << action;
    const uint64_t changed = flips_bb(own, opp, placed) | placed;
    BB out;
    if (player > 0) { out.black = b.black |  changed; out.white = b.white & ~changed; }
    else            { out.white = b.white |  changed; out.black = b.black & ~changed; }
    return out;
}

/* board_batched.legal_action_masks for one game: squares, plus PASS legal
   exactly when no square is. */
inline void write_action_mask(uint64_t moves, float* row) {
    for (int i = 0; i < NSQ; ++i) row[i] = ((moves >> i) & 1ULL) ? 1.0f : 0.0f;
    row[PASS] = (moves == 0ULL) ? 1.0f : 0.0f;
}

// --- numpy plumbing ---------------------------------------------------------

using ArrI8  = py::array_t<int8_t,  py::array::c_style | py::array::forcecast>;
using ArrI64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using ArrF32 = py::array_t<float,   py::array::c_style | py::array::forcecast>;

const int8_t* boards_ptr(const ArrI8& a, py::ssize_t& B) {
    if (a.ndim() != 3 || a.shape(1) != BOARD_N || a.shape(2) != BOARD_N)
        throw std::invalid_argument("boards must have shape [B, 8, 8]");
    B = a.shape(0);
    return a.data();
}

const int8_t* players_ptr(const ArrI8& a, py::ssize_t B) {
    if (a.ndim() != 1 || a.shape(0) != B)
        throw std::invalid_argument("players must have shape [B] matching boards");
    return a.data();
}

// --- exported engine functions ----------------------------------------------

py::array_t<bool> legal_move_masks(ArrI8 boards, ArrI8 players) {
    py::ssize_t B;
    const int8_t* bp = boards_ptr(boards, B);
    const int8_t* pp = players_ptr(players, B);
    py::array_t<bool> out({B, py::ssize_t(BOARD_N), py::ssize_t(BOARD_N)});
    bool* op = out.mutable_data();
    for (py::ssize_t g = 0; g < B; ++g) {
        uint64_t own, opp;
        own_opp(unpack(bp + g * NSQ), pp[g], own, opp);
        const uint64_t m = legal_moves_bb(own, opp);
        bool* row = op + g * NSQ;
        for (int i = 0; i < NSQ; ++i) row[i] = ((m >> i) & 1ULL) != 0ULL;
    }
    return out;
}

py::array_t<float> legal_action_masks(ArrI8 boards, ArrI8 players) {
    py::ssize_t B;
    const int8_t* bp = boards_ptr(boards, B);
    const int8_t* pp = players_ptr(players, B);
    py::array_t<float> out({B, py::ssize_t(POLICY_SIZE)});
    float* op = out.mutable_data();
    for (py::ssize_t g = 0; g < B; ++g) {
        uint64_t own, opp;
        own_opp(unpack(bp + g * NSQ), pp[g], own, opp);
        write_action_mask(legal_moves_bb(own, opp), op + g * POLICY_SIZE);
    }
    return out;
}

py::array_t<int8_t> apply_moves(ArrI8 boards, ArrI8 players, ArrI64 actions) {
    py::ssize_t B;
    const int8_t* bp = boards_ptr(boards, B);
    const int8_t* pp = players_ptr(players, B);
    if (actions.ndim() != 1 || actions.shape(0) != B)
        throw std::invalid_argument("actions must have shape [B] matching boards");
    const int64_t* ap = actions.data();
    py::array_t<int8_t> out({B, py::ssize_t(BOARD_N), py::ssize_t(BOARD_N)});
    int8_t* op = out.mutable_data();
    for (py::ssize_t g = 0; g < B; ++g)
        pack(apply_bb(unpack(bp + g * NSQ), pp[g], ap[g]), op + g * NSQ);
    return out;
}

py::array_t<bool> is_terminal(ArrI8 boards) {
    py::ssize_t B;
    const int8_t* bp = boards_ptr(boards, B);
    py::array_t<bool> out(B);
    bool* op = out.mutable_data();
    for (py::ssize_t g = 0; g < B; ++g) op[g] = terminal_bb(unpack(bp + g * NSQ));
    return out;
}

std::tuple<py::array_t<int64_t>, py::array_t<int64_t>> count_discs(ArrI8 boards) {
    py::ssize_t B;
    const int8_t* bp = boards_ptr(boards, B);
    py::array_t<int64_t> black(B), white(B);
    int64_t* bd = black.mutable_data();
    int64_t* wd = white.mutable_data();
    for (py::ssize_t g = 0; g < B; ++g) {
        const BB b = unpack(bp + g * NSQ);
        bd[g] = popcount64(b.black);
        wd[g] = popcount64(b.white);
    }
    return std::make_tuple(black, white);
}

py::array_t<int8_t> winner(ArrI8 boards) {
    py::ssize_t B;
    const int8_t* bp = boards_ptr(boards, B);
    py::array_t<int8_t> out(B);
    int8_t* op = out.mutable_data();
    for (py::ssize_t g = 0; g < B; ++g) op[g] = int8_t(winner_bb(unpack(bp + g * NSQ)));
    return out;
}

/* encode.py's 3 planes in the side-to-move's POV: own discs, opponent discs,
   own legal-move mask. */
py::array_t<float> encode_batch(ArrI8 boards, ArrI8 players) {
    py::ssize_t B;
    const int8_t* bp = boards_ptr(boards, B);
    const int8_t* pp = players_ptr(players, B);
    py::array_t<float> out({B, py::ssize_t(3), py::ssize_t(BOARD_N), py::ssize_t(BOARD_N)});
    float* op = out.mutable_data();
    for (py::ssize_t g = 0; g < B; ++g) {
        uint64_t own, opp;
        own_opp(unpack(bp + g * NSQ), pp[g], own, opp);
        const uint64_t legal = legal_moves_bb(own, opp);
        float* base = op + g * 3 * NSQ;
        for (int i = 0; i < NSQ; ++i) {
            const uint64_t m = 1ULL << i;
            base[i]             = (own   & m) ? 1.0f : 0.0f;
            base[NSQ + i]       = (opp   & m) ? 1.0f : 0.0f;
            base[2 * NSQ + i]   = (legal & m) ? 1.0f : 0.0f;
        }
    }
    return out;
}

// --- PUCT search (the mcts_batched.py port) ---------------------------------

/* mcts_batched._select_actions for one node, op for op:

       sum_n  = N.sum()
       q      = W/N where N>0 else 0
       u      = c_puct * P * sqrt(sum_n) / (1 + N)
       scores = (sum_n == 0) ? P : q + u
       scores = legal ? scores : -1e30
       return scores.argmax()          # FIRST maximum

   sum_n is exact in float32 (N holds integers), so the naive accumulation here
   matches NumPy's pairwise summation bit-for-bit. */
inline int select_action(const float* P, const float* N, const float* W,
                         const uint8_t* legal, float c_puct) {
    float sum_n = 0.0f;
    for (int j = 0; j < POLICY_SIZE; ++j) sum_n += N[j];
    const bool fresh = (sum_n == 0.0f);
    const float sqrt_sum = std::sqrt(sum_n);

    int best = 0;
    float best_score = 0.0f;
    for (int j = 0; j < POLICY_SIZE; ++j) {
        float score;
        if (fresh) {
            score = P[j];
        } else {
            const float q = (N[j] > 0.0f) ? (W[j] / N[j]) : 0.0f;
            const float u = c_puct * P[j] * sqrt_sum / (1.0f + N[j]);
            score = q + u;
        }
        if (!legal[j]) score = NEG_INF;
        if (j == 0 || score > best_score) { best_score = score; best = j; }
    }
    return best;
}

std::tuple<py::array_t<float>, py::array_t<float>> run_batched(
        ArrI8 boards_in, ArrI8 players_in, int sims, py::function evaluate,
        double c_puct_in, double dirichlet_eps_in, py::object noise_obj) {

    py::ssize_t B;
    const int8_t* bp = boards_ptr(boards_in, B);
    const int8_t* pp = players_ptr(players_in, B);
    if (sims < 0) throw std::invalid_argument("sims must be >= 0");

    const float c_puct = float(c_puct_in);
    // NumPy evaluates the Python-scalar expression `1.0 - eps` in float64 and only
    // then casts to float32 for the array multiply — mirror that exactly.
    const float keep = float(1.0 - dirichlet_eps_in);
    const float eps = float(dirichlet_eps_in);

    const int max_nodes = sims + 1;                 // root + one new node per sim
    const size_t nodes_total = size_t(B) * max_nodes;
    const size_t rows_total = nodes_total * POLICY_SIZE;

    std::vector<float> P(rows_total, 0.0f), N(rows_total, 0.0f), W(rows_total, 0.0f);
    std::vector<uint8_t> legal(rows_total, 0);
    std::vector<int32_t> children(rows_total, -1);
    std::vector<uint64_t> nb_black(nodes_total, 0ULL), nb_white(nodes_total, 0ULL);
    std::vector<int8_t> node_player(nodes_total, 0);
    std::vector<uint8_t> node_terminal(nodes_total, 0);
    std::vector<float> node_tvalue(nodes_total, 0.0f);
    std::vector<int32_t> n_nodes(size_t(B), 1);     // node 0 (the root) is allocated

    // Pending leaf evaluations for the current step, in ASCENDING game order —
    // the same batch composition and order the NumPy path produces, so the net
    // sees identical batches and rounds identically.
    std::vector<int32_t> ev_game, ev_node;
    std::vector<BB> ev_board;
    std::vector<int8_t> ev_player;
    std::vector<float> ev_value;
    std::vector<size_t> ev_live;

    /* mcts_batched._eval_into: terminal leaves take their exact game result, the
       rest go through ONE batched callback into Python. */
    auto eval_into = [&]() {
        const size_t k = ev_game.size();
        ev_value.assign(k, 0.0f);
        if (k == 0) return;
        ev_live.clear();
        for (size_t i = 0; i < k; ++i) {
            const size_t nid = size_t(ev_game[i]) * max_nodes + ev_node[i];
            if (terminal_bb(ev_board[i])) {
                const int w = winner_bb(ev_board[i]);
                const float tv = (w == 0) ? 0.0f : ((w == int(ev_player[i])) ? 1.0f : -1.0f);
                node_terminal[nid] = 1;
                node_tvalue[nid] = tv;
                ev_value[i] = tv;
            } else {
                ev_live.push_back(i);
            }
        }
        if (ev_live.empty()) return;

        const py::ssize_t k_live = py::ssize_t(ev_live.size());
        py::array_t<int8_t> bat({k_live, py::ssize_t(BOARD_N), py::ssize_t(BOARD_N)});
        py::array_t<int8_t> pat(k_live);
        int8_t* bd = bat.mutable_data();
        int8_t* pd = pat.mutable_data();
        for (py::ssize_t i = 0; i < k_live; ++i) {
            pack(ev_board[ev_live[i]], bd + i * NSQ);
            pd[i] = ev_player[ev_live[i]];
        }

        py::object res = evaluate(bat, pat);
        py::tuple t = res.cast<py::tuple>();
        if (t.size() != 2)
            throw std::runtime_error("evaluate() must return (priors, values)");
        ArrF32 priors = t[0].cast<ArrF32>();
        ArrF32 values = t[1].cast<ArrF32>();
        if (priors.ndim() != 2 || priors.shape(0) != k_live || priors.shape(1) != POLICY_SIZE)
            throw std::runtime_error("evaluate() priors must have shape [k, 65]");
        if (values.ndim() != 1 || values.shape(0) != k_live)
            throw std::runtime_error("evaluate() values must have shape [k]");
        const float* pr = priors.data();
        const float* vl = values.data();

        for (py::ssize_t i = 0; i < k_live; ++i) {
            const size_t idx = ev_live[i];
            const size_t nid = size_t(ev_game[idx]) * max_nodes + ev_node[idx];
            std::memcpy(&P[nid * POLICY_SIZE], pr + size_t(i) * POLICY_SIZE,
                        POLICY_SIZE * sizeof(float));
            uint64_t own, opp;
            own_opp(ev_board[idx], ev_player[idx], own, opp);
            const uint64_t m = legal_moves_bb(own, opp);
            uint8_t* lrow = &legal[nid * POLICY_SIZE];
            for (int j = 0; j < NSQ; ++j) lrow[j] = ((m >> j) & 1ULL) ? 1 : 0;
            lrow[PASS] = (m == 0ULL) ? 1 : 0;
            ev_value[idx] = vl[i];
        }
    };

    // --- root ---
    for (py::ssize_t g = 0; g < B; ++g) {
        const BB b = unpack(bp + g * NSQ);
        const size_t nid = size_t(g) * max_nodes;
        nb_black[nid] = b.black;
        nb_white[nid] = b.white;
        node_player[nid] = pp[g];
        ev_game.push_back(int32_t(g));
        ev_node.push_back(0);
        ev_board.push_back(b);
        ev_player.push_back(pp[g]);
    }
    eval_into();

    // Root Dirichlet noise, already drawn and normalised on the Python side
    // (mcts_batched._add_dirichlet). Terminal roots have no legal actions, so
    // their priors are left alone — same as NumPy.
    if (!noise_obj.is_none()) {
        ArrF32 noise = noise_obj.cast<ArrF32>();
        if (noise.ndim() != 2 || noise.shape(0) != B || noise.shape(1) != POLICY_SIZE)
            throw std::invalid_argument("noise must have shape [B, 65]");
        const float* nz = noise.data();
        for (py::ssize_t g = 0; g < B; ++g) {
            const size_t nid = size_t(g) * max_nodes;
            float* prow = &P[nid * POLICY_SIZE];
            const uint8_t* lrow = &legal[nid * POLICY_SIZE];
            const float* nrow = nz + size_t(g) * POLICY_SIZE;
            for (int j = 0; j < POLICY_SIZE; ++j)
                if (lrow[j]) prow[j] = keep * prow[j] + eps * nrow[j];
        }
    }

    // --- simulations ---
    std::vector<int32_t> path_nodes(size_t(B) * max_nodes);
    std::vector<int32_t> path_actions(size_t(B) * max_nodes);
    std::vector<int32_t> plen(size_t(B), 0);
    std::vector<float> leaf_value(size_t(B), 0.0f);
    std::vector<int8_t> leaf_player(size_t(B), 0);

    for (int s = 0; s < sims; ++s) {
        ev_game.clear(); ev_node.clear(); ev_board.clear(); ev_player.clear();

        // SELECT: descend each tree to a leaf. Trees are independent, so doing
        // them one at a time here is identical to NumPy's lockstep descent.
        for (py::ssize_t g = 0; g < B; ++g) {
            int32_t cur = 0, len = 0;
            int32_t* pn = &path_nodes[size_t(g) * max_nodes];
            int32_t* pa = &path_actions[size_t(g) * max_nodes];
            while (true) {
                const size_t nid = size_t(g) * max_nodes + cur;
                if (node_terminal[nid]) {            // terminal leaf: exact value
                    leaf_value[g] = node_tvalue[nid];
                    leaf_player[g] = node_player[nid];
                    break;
                }
                const int a = select_action(&P[nid * POLICY_SIZE], &N[nid * POLICY_SIZE],
                                            &W[nid * POLICY_SIZE], &legal[nid * POLICY_SIZE],
                                            c_puct);
                pn[len] = cur;                       // the edge is on the path even
                pa[len] = int32_t(a);                // when it leads to a new node
                ++len;
                const int32_t child = children[nid * POLICY_SIZE + a];
                if (child < 0) {                     // EXPAND one new node
                    const int32_t new_idx = n_nodes[size_t(g)]++;
                    children[nid * POLICY_SIZE + a] = new_idx;
                    const BB parent{nb_black[nid], nb_white[nid]};
                    const int8_t parent_player = node_player[nid];
                    const BB childb = apply_bb(parent, parent_player, a);
                    const int8_t childp = int8_t(-parent_player);   // a move OR a pass flips
                    const size_t cid = size_t(g) * max_nodes + new_idx;
                    nb_black[cid] = childb.black;
                    nb_white[cid] = childb.white;
                    node_player[cid] = childp;
                    ev_game.push_back(int32_t(g));
                    ev_node.push_back(new_idx);
                    ev_board.push_back(childb);
                    ev_player.push_back(childp);
                    leaf_player[g] = childp;
                    break;
                }
                cur = child;
            }
            plen[size_t(g)] = len;
        }

        // EVALUATE the new leaves — one batched call back into Python.
        eval_into();
        for (size_t i = 0; i < ev_game.size(); ++i)
            leaf_value[size_t(ev_game[i])] = ev_value[i];

        // BACKUP: +v where the node's mover matches the leaf's mover, else -v.
        // Players are compared explicitly, so passes stay correct.
        for (py::ssize_t g = 0; g < B; ++g) {
            const float lv = leaf_value[size_t(g)];
            const int8_t lp = leaf_player[size_t(g)];
            const int32_t* pn = &path_nodes[size_t(g) * max_nodes];
            const int32_t* pa = &path_actions[size_t(g) * max_nodes];
            for (int32_t l = 0; l < plen[size_t(g)]; ++l) {
                const size_t nid = size_t(g) * max_nodes + pn[l];
                const float sign = (node_player[nid] == lp) ? 1.0f : -1.0f;
                N[nid * POLICY_SIZE + pa[l]] += 1.0f;
                W[nid * POLICY_SIZE + pa[l]] += sign * lv;
            }
        }
    }

    // Return the root's raw N and W rows. root_value is finished off in NumPy on
    // the Python side (W.sum(1)/N.sum(1)) because W's sum IS order-sensitive —
    // letting NumPy do it keeps that last reduction bit-identical for free.
    py::array_t<float> counts({B, py::ssize_t(POLICY_SIZE)});
    py::array_t<float> wroot({B, py::ssize_t(POLICY_SIZE)});
    float* cd = counts.mutable_data();
    float* wd = wroot.mutable_data();
    for (py::ssize_t g = 0; g < B; ++g) {
        const size_t nid = size_t(g) * max_nodes;
        std::memcpy(cd + size_t(g) * POLICY_SIZE, &N[nid * POLICY_SIZE],
                    POLICY_SIZE * sizeof(float));
        std::memcpy(wd + size_t(g) * POLICY_SIZE, &W[nid * POLICY_SIZE],
                    POLICY_SIZE * sizeof(float));
    }
    return std::make_tuple(counts, wroot);
}

}  // namespace

PYBIND11_MODULE(othello_native, m) {
    m.doc() = "C++ Othello engine (bitboards) + batched PUCT MCTS — see the "
              "module header in native/othello_native.cpp";

    // Bumped whenever the Python<->C++ signatures change, so engine/native.py can
    // refuse a stale .so instead of failing mysteriously at call time.
    m.attr("__abi__") = 1;
    m.attr("__version__") = "1.0";

    m.def("legal_move_masks", &legal_move_masks, py::arg("boards"), py::arg("players"));
    m.def("legal_action_masks", &legal_action_masks, py::arg("boards"), py::arg("players"));
    m.def("apply_moves", &apply_moves, py::arg("boards"), py::arg("players"), py::arg("actions"));
    m.def("is_terminal", &is_terminal, py::arg("boards"));
    m.def("count_discs", &count_discs, py::arg("boards"));
    m.def("winner", &winner, py::arg("boards"));
    m.def("encode_batch", &encode_batch, py::arg("boards"), py::arg("players"));

    m.def("run_batched", &run_batched,
          py::arg("boards"), py::arg("players"), py::arg("sims"), py::arg("evaluate"),
          py::arg("c_puct"), py::arg("dirichlet_eps"), py::arg("noise"),
          "B PUCT trees searched with one batched `evaluate` callback per step. "
          "Returns (root_visit_counts [B,65], root_W [B,65]).");
}
