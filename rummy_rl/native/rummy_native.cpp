// Rummy engine in C++: hand scoring, the game itself, what each player can see,
// and two baseline bots (random and greedy). Rules follow RULES.md; rule IDs are
// quoted where they are implemented.
//
// Cards are ints 0..51: suit * 13 + rank, rank 0 = A .. 12 = K, suits S H D C.
// A set of cards is a 64-bit mask with bit `card` set. Same encoding as
// engine/hand.py, which is the reference the scorer here is tested against.
//
// Many games run side by side in one Env so Python pays its per-call cost once
// per batch, not once per game.
//
// Actions (one number per move):
//   0..51  throw that card (DISCARD phase)
//   52     draw from the draw pile (DRAW phase)
//   53     take the top card of the discard pile (DRAW phase)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using u64 = uint64_t;

static constexpr int N_CARDS = 52;
static constexpr int HAND = 13;
static constexpr int ACT_STOCK = 52;
static constexpr int ACT_PILE = 53;
static constexpr int N_ACTIONS = 54;
static constexpr int N_PLANES = 10;
static constexpr int N_SCALARS = 6;
static constexpr float VALUE_SCALE = 130.f;  // most points a hand can cost (13 cards x 10)
static constexpr u64 ALL_CARDS = (u64(1) << N_CARDS) - 1;

static inline int card_value(int c) {  // R6.3
    int r = c % 13;
    return (r == 0 || r >= 9) ? 10 : r + 1;
}
static inline u64 bit(int c) { return u64(1) << c; }
static inline int popcount(u64 m) { return __builtin_popcountll(m); }
static inline int lowest(u64 m) { return __builtin_ctzll(m); }

// --------------------------------------------------------------------------- //
// Scorer (RULES.md sections 5 and 6)
//
// Same search as engine/hand.py: list every meld the hand contains (runs of
// 3-6, sets of 3-4), then find the best cover by always deciding the lowest
// card left (leave it as a leftover, or put it in one of its melds), memoised.
// The mandatory run of 4+ is placed first. Runs of 7+ never need listing: they
// split into a 4+ run and a 3+ run covering the same cards. A 6-run does not
// (6 = 3+3 loses the 4+ run), so it is listed.
//
// Inside the solver, cards are renumbered 0..n-1 (n <= 14) so a sub-hand is a
// 14-bit number and the memo is a flat array.
// --------------------------------------------------------------------------- //

struct Meld {
    uint16_t mask;
    int16_t value;
};

struct Solver {
    int n = 0;
    int card_at[14];
    int value_at[14];
    std::vector<Meld> melds, pure;         // pure = runs of 4+
    std::vector<int> by_idx[14];           // melds containing each card index
    int16_t memo[1 << 14];
    uint32_t stamp[1 << 14] = {};
    uint32_t gen = 0;

    void build(u64 hand) {
        n = popcount(hand);
        if (n > 14) throw std::runtime_error("solver: more than 14 cards");
        int idx_of[N_CARDS];
        int k = 0;
        for (u64 m = hand; m; m &= m - 1) {
            int c = lowest(m);
            idx_of[c] = k;
            card_at[k] = c;
            value_at[k] = card_value(c);
            ++k;
        }
        melds.clear();
        pure.clear();
        for (int i = 0; i < 14; ++i) by_idx[i].clear();

        // Runs (R5.1, R5.3): position p in 0..13 is rank p % 13; 13 is the high ace.
        for (int s = 0; s < 4; ++s)
            for (int len = 3; len <= 6; ++len)
                for (int start = 0; start + len <= 14; ++start) {
                    uint16_t m = 0;
                    int v = 0;
                    bool ok = true;
                    for (int p = start; p < start + len; ++p) {
                        int c = s * 13 + p % 13;
                        if (!(hand & bit(c))) { ok = false; break; }
                        m |= uint16_t(1) << idx_of[c];
                        v += card_value(c);
                    }
                    if (!ok) continue;
                    melds.push_back({m, int16_t(v)});
                    if (len >= 4) pure.push_back({m, int16_t(v)});
                }
        // Sets (R5.2): 3 or 4 of one rank.
        for (int r = 0; r < 13; ++r) {
            int idx[4], cnt = 0;
            for (int s = 0; s < 4; ++s)
                if (hand & bit(s * 13 + r)) idx[cnt++] = idx_of[s * 13 + r];
            if (cnt < 3) continue;
            int v = card_value(r);
            if (cnt == 4) {
                melds.push_back({uint16_t((1 << idx[0]) | (1 << idx[1]) | (1 << idx[2]) | (1 << idx[3])), int16_t(4 * v)});
                for (int skip = 0; skip < 4; ++skip) {
                    uint16_t m = 0;
                    for (int j = 0; j < 4; ++j)
                        if (j != skip) m |= uint16_t(1) << idx[j];
                    melds.push_back({m, int16_t(3 * v)});
                }
            } else {
                melds.push_back({uint16_t((1 << idx[0]) | (1 << idx[1]) | (1 << idx[2])), int16_t(3 * v)});
            }
        }
        for (int i = 0; i < (int)melds.size(); ++i)
            for (uint16_t m = melds[i].mask; m; m &= m - 1) by_idx[__builtin_ctz(m)].push_back(i);

        if (++gen == 0) {  // stamp counter wrapped: forget everything
            std::fill(std::begin(stamp), std::end(stamp), 0u);
            gen = 1;
        }
    }

    int total(uint16_t sub) const {
        int t = 0;
        for (uint16_t m = sub; m; m &= m - 1) t += value_at[__builtin_ctz(m)];
        return t;
    }

    // Max value of `sub` coverable by disjoint melds (no 4+ run requirement).
    int covered(uint16_t sub) {
        if (!sub) return 0;
        if (stamp[sub] == gen) return memo[sub];
        int i = __builtin_ctz(sub);
        int best = covered(sub & (sub - 1));  // lowest card stays a leftover
        for (int mi : by_idx[i]) {
            uint16_t m = melds[mi].mask;
            if ((m & sub) == m) {
                int v = melds[mi].value + covered(sub ^ m);
                if (v > best) best = v;
            }
        }
        stamp[sub] = gen;
        memo[sub] = int16_t(best);
        return best;
    }

    // Minimum score of the cards in `sub` (R6.4).
    int score(uint16_t sub) {
        int best = -1;
        for (const Meld& p : pure)
            if ((p.mask & sub) == p.mask) best = std::max(best, p.value + covered(sub ^ p.mask));
        int t = total(sub);
        return best < 0 ? t : t - best;
    }

    uint16_t full() const { return uint16_t((1u << n) - 1); }
};

static thread_local Solver SOLVER;

static int score13(u64 hand) {
    SOLVER.build(hand);
    return SOLVER.score(SOLVER.full());
}

// Score of the 13 cards left after throwing each card of a 14-card hand.
// Fills cards[i], scores[i] in ascending card order.
static void discard_scores(u64 hand, int cards[14], int scores[14]) {
    SOLVER.build(hand);
    uint16_t full = SOLVER.full();
    for (int i = 0; i < SOLVER.n; ++i) {
        cards[i] = SOLVER.card_at[i];
        scores[i] = SOLVER.score(uint16_t(full ^ (1u << i)));
    }
}

// Best discard: lowest score; ties keep the highest-value card to throw
// (same tie rule as engine/hand.py best_discard).
static void best_discard(u64 hand, int& best_card, int& best_score) {
    int cards[14], scores[14];
    discard_scores(hand, cards, scores);
    best_card = -1;
    best_score = 1 << 30;
    for (int i = 0; i < SOLVER.n; ++i)
        if (scores[i] < best_score || (scores[i] == best_score && card_value(cards[i]) > card_value(best_card))) {
            best_score = scores[i];
            best_card = cards[i];
        }
}

// --------------------------------------------------------------------------- //
// Random numbers: splitmix64, one stream per game so any hand can be replayed.
// --------------------------------------------------------------------------- //

static inline u64 splitmix(u64& s) {
    u64 z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline int below(u64& s, int k) {  // uniform in [0, k)
    return int((unsigned __int128)splitmix(s) * unsigned(k) >> 64);
}

// --------------------------------------------------------------------------- //
// Game (RULES.md sections 2-4)
// --------------------------------------------------------------------------- //

enum Phase : int { DRAW = 0, DISCARD = 1, OVER = 2 };

struct Game {
    u64 hand[2];
    uint8_t stock[N_CARDS];  // draw pile, top = stock[n_stock - 1]
    int n_stock;
    uint8_t pile[N_CARDS];   // discard pile, top = pile[n_pile - 1]
    int n_pile;
    u64 pile_mask;
    int to_move, phase, turn, reshuffles;
    int drawn;                // card drawn this turn, -1 if none
    bool drew_from_pile;
    // What everyone can see (R8.2)
    u64 held_known[2];        // cards p took from the pile and still holds
    u64 discarded_by[2];      // cards p threw this hand
    u64 taken_by[2];          // cards p took from the pile this hand
    u64 recycled;             // cards shuffled back into the draw pile
    u64 last_recycled;        // cards that went into the draw pile at the latest reshuffle
    // Result
    int winner;               // -1 while playing, or when the turn cap ends the hand
    int points;               // loser's score (R6.2)
    int declare_card;         // card thrown when declaring, -1 if none
    bool capped;
    int cap_score[2];         // T9.1: what each player would pay if caught when the turn cap hit
    u64 rng;
};

static void push_pile(Game& g, int c) {
    g.pile[g.n_pile++] = uint8_t(c);
    g.pile_mask |= bit(c);
}

static void reset_game(Game& g, u64 seed, int first) {
    g = Game{};
    g.rng = seed;
    uint8_t deck[N_CARDS];
    for (int i = 0; i < N_CARDS; ++i) deck[i] = uint8_t(i);
    for (int i = N_CARDS - 1; i > 0; --i) std::swap(deck[i], deck[below(g.rng, i + 1)]);  // R2.1
    for (int i = 0; i < 2 * HAND; ++i) g.hand[i & 1] |= bit(deck[i]);
    push_pile(g, deck[2 * HAND]);                                                       // R2.2
    g.n_stock = 0;
    for (int i = N_CARDS - 1; i > 2 * HAND; --i) g.stock[g.n_stock++] = deck[i];        // R2.3
    g.to_move = first;
    g.phase = DRAW;
    g.drawn = -1;
    g.winner = -1;
    g.declare_card = -1;
}

// R4: everything under the top card of the discard pile becomes the new draw pile.
static void reshuffle(Game& g) {
    int top = g.pile[g.n_pile - 1];
    g.n_stock = 0;
    g.last_recycled = 0;
    for (int i = 0; i < g.n_pile - 1; ++i) {
        g.stock[g.n_stock++] = g.pile[i];
        g.recycled |= bit(g.pile[i]);
        g.last_recycled |= bit(g.pile[i]);
    }
    for (int i = g.n_stock - 1; i > 0; --i) std::swap(g.stock[i], g.stock[below(g.rng, i + 1)]);
    g.pile[0] = uint8_t(top);
    g.n_pile = 1;
    g.pile_mask = bit(top);
    g.reshuffles++;
}

static void throw_card(Game& g, int p, int c) {
    g.hand[p] &= ~bit(c);
    push_pile(g, c);
    g.discarded_by[p] |= bit(c);
    g.held_known[p] &= ~bit(c);
}

static std::string legal_error(const Game& g, int a) {
    if (g.phase == OVER) return "the hand is over";
    if (g.phase == DRAW) {
        if (a == ACT_STOCK || a == ACT_PILE) return "";
        return "expected a draw (52 = draw pile, 53 = discard pile), got " + std::to_string(a);
    }
    if (a < 0 || a >= N_CARDS || !(g.hand[g.to_move] & bit(a)))
        return "expected a card in hand to throw, got " + std::to_string(a);
    return "";
}

// Applies one legal action. Returns true if the hand ended on this action.
static bool apply_action(Game& g, int a, int turn_cap) {
    int p = g.to_move;
    if (g.phase == DRAW) {  // R3.1
        int c;
        if (a == ACT_PILE) {
            c = g.pile[--g.n_pile];
            g.pile_mask &= ~bit(c);
            g.taken_by[p] |= bit(c);
            g.held_known[p] |= bit(c);
        } else {
            if (g.n_stock == 0) reshuffle(g);
            c = g.stock[--g.n_stock];
        }
        g.hand[p] |= bit(c);
        g.drawn = c;
        g.drew_from_pile = (a == ACT_PILE);

        // Automatic declare (R3.4): declaring is always best, so if any discard
        // leaves a valid hand, throw it and end the hand.
        int d, s;
        best_discard(g.hand[p], d, s);
        if (s == 0) {
            throw_card(g, p, d);
            g.declare_card = d;
            g.winner = p;
            g.points = score13(g.hand[1 - p]);  // R6.2
            g.phase = OVER;
            return true;
        }
        g.phase = DISCARD;
        return false;
    }
    throw_card(g, p, a);  // R3.2
    g.drawn = -1;
    g.drew_from_pile = false;
    g.to_move = 1 - p;
    g.phase = DRAW;
    g.turn++;
    if (turn_cap > 0 && g.turn >= turn_cap) {  // T9.1: score both hands as if caught
        g.phase = OVER;
        g.capped = true;
        g.cap_score[0] = score13(g.hand[0]);
        g.cap_score[1] = score13(g.hand[1]);
        return true;
    }
    return false;
}

// Points player p gains (+) or loses (-) from a finished hand (T9.2). A hand
// ended by the turn cap scores both players as if caught (T9.1).
static float hand_reward(const Game& g, int p) {
    if (g.capped) return float(g.cap_score[1 - p] - g.cap_score[p]);
    if (g.winner < 0) return 0.f;
    return g.winner == p ? float(g.points) : -float(g.points);
}

// Every rule the state must obey. Returns "" when all hold.
static std::string validate_game(const Game& g) {
    u64 stock_mask = 0, pile_mask = 0;
    for (int i = 0; i < g.n_stock; ++i) stock_mask |= bit(g.stock[i]);
    for (int i = 0; i < g.n_pile; ++i) pile_mask |= bit(g.pile[i]);
    if (popcount(stock_mask) != g.n_stock) return "duplicate card in the draw pile";
    if (popcount(pile_mask) != g.n_pile) return "duplicate card in the discard pile";
    if (g.reshuffles && (stock_mask & ~g.last_recycled)) return "draw pile holds a card that wasn't reshuffled into it";
    if (pile_mask != g.pile_mask) return "discard pile mask out of date";
    u64 parts[4] = {g.hand[0], g.hand[1], stock_mask, pile_mask};
    u64 seen = 0;
    int count = 0;
    for (u64 m : parts) {
        if (seen & m) return "a card is in two places";
        seen |= m;
        count += popcount(m);
    }
    if (seen != ALL_CARDS || count != N_CARDS) return "cards missing: expected all 52 exactly once";
    int h0 = popcount(g.hand[0]), h1 = popcount(g.hand[1]);
    int mover = popcount(g.hand[g.to_move]), other = popcount(g.hand[1 - g.to_move]);
    if (g.phase == DRAW && (mover != HAND || other != HAND)) return "hands must hold 13 before a draw";
    if (g.phase == DISCARD && (mover != HAND + 1 || other != HAND)) return "mover must hold 14 after a draw";
    if (g.phase == OVER && (h0 != HAND || h1 != HAND)) return "hands must hold 13 when the hand ends";
    if (g.phase == DRAW && g.n_pile < 1) return "discard pile empty before a draw";
    for (int p = 0; p < 2; ++p)
        if (g.held_known[p] & ~g.hand[p]) return "known-held card not in that hand";
    if (g.phase == DISCARD && (g.drawn < 0 || !(g.hand[g.to_move] & bit(g.drawn)))) return "drawn card not in hand";
    if (g.phase == OVER && !g.capped) {
        if (g.winner < 0) return "hand over without a winner or cap";
        if (score13(g.hand[g.winner]) != 0) return "winner's hand is not a valid declare";
        if (g.points != score13(g.hand[1 - g.winner])) return "loser's points are wrong";
        if (g.pile[g.n_pile - 1] != g.declare_card) return "declare card is not on top of the pile";
    }
    if (g.phase == OVER && g.capped && g.winner != -1) return "capped hand has a winner";
    if (g.phase == OVER && g.capped && (g.cap_score[0] != score13(g.hand[0]) || g.cap_score[1] != score13(g.hand[1])))
        return "capped hand scored wrongly";
    return "";
}

// --------------------------------------------------------------------------- //
// Bots
// --------------------------------------------------------------------------- //

// How many cards in `hand` could join card c in a meld soon: same rank counts 2,
// same suit one rank away counts 2, two ranks away counts 1 (ace is next to 2 and K).
static int potential(int c, u64 hand) {
    int r = c % 13, s = c / 13, pts = 0;
    for (u64 m = hand & ~bit(c); m; m &= m - 1) {
        int d = lowest(m);
        if (d % 13 == r) { pts += 2; continue; }
        if (d / 13 != s) continue;
        int gap = std::abs(d % 13 - r);
        gap = std::min(gap, 13 - gap);  // K-A counts as adjacent (no wrap past A in melds, but close enough)
        if (gap == 1) pts += 2;
        else if (gap == 2) pts += 1;
    }
    return pts;
}

// Greedy: keep the hand as cheap as possible right now. It only looks at its own
// hand and the top of the discard pile, like a human player.
//   Draw:  take the top discard only if it lowers the score it would pay (with
//          the best throw afterwards, not counting throwing that card back).
//   Throw: the card whose loss leaves the lowest score; ties throw the card with
//          the fewest partners, then the higher value.
static int greedy_action(const Game& g) {
    int p = g.to_move;
    if (g.phase == DRAW) {
        int now = score13(g.hand[p]);
        int top = g.pile[g.n_pile - 1];
        int cards[14], scores[14];
        discard_scores(g.hand[p] | bit(top), cards, scores);
        int with = 1 << 30;
        for (int i = 0; i < 14; ++i)
            if (cards[i] != top) with = std::min(with, scores[i]);
        return with < now ? ACT_PILE : ACT_STOCK;
    }
    int cards[14], scores[14];
    discard_scores(g.hand[p], cards, scores);
    int best = -1;
    int best_key[3] = {0, 0, 0};
    for (int i = 0; i < 14; ++i) {
        int key[3] = {scores[i], potential(cards[i], g.hand[p]), -card_value(cards[i])};
        if (best < 0 || std::lexicographical_compare(key, key + 3, best_key, best_key + 3)) {
            best = cards[i];
            std::copy(key, key + 3, best_key);
        }
    }
    return best;
}

static int random_action(const Game& g, u64& rng) {
    if (g.phase == DRAW) return below(rng, 2) ? ACT_PILE : ACT_STOCK;
    u64 h = g.hand[g.to_move];
    int k = below(rng, popcount(h));
    for (; k > 0; --k) h &= h - 1;
    return lowest(h);
}

// --------------------------------------------------------------------------- //
// What a player sees (R8)
// --------------------------------------------------------------------------- //

// Observation of game g for the player to move:
//   planes [10][52]: 0 my hand, 1 top of the discard pile, 2 whole discard pile,
//       3 cards the opponent is known to hold, 4 cards the opponent threw,
//       5 cards the opponent took from the pile, 6 cards I threw,
//       7 cards that went into the draw pile at the latest reshuffle,
//       8 cards I can't locate, 9 the card I just drew (discard phase only)
//   scalars [6]: draw phase?, draw pile size / 52, discard pile size / 52,
//       turn / 200, reshuffles, drew from the discard pile this turn?
//   legal [54] (optional): which actions are allowed.
static void fill_obs(const Game& g, float* P, float* S, bool* legal) {
    int p = g.to_move, o = 1 - p;
    u64 top = g.n_pile ? bit(g.pile[g.n_pile - 1]) : 0;
    u64 drawn = (g.phase == DISCARD && g.drawn >= 0) ? bit(g.drawn) : 0;
    u64 masks[N_PLANES] = {
        g.hand[p], top, g.pile_mask, g.held_known[o], g.discarded_by[o], g.taken_by[o],
        g.discarded_by[p], g.last_recycled, ALL_CARDS & ~(g.hand[p] | g.pile_mask | g.held_known[o]), drawn,
    };
    for (int k = 0; k < N_PLANES; ++k)
        for (int c = 0; c < N_CARDS; ++c) P[k * N_CARDS + c] = (masks[k] >> c & 1) ? 1.f : 0.f;
    S[0] = g.phase == DRAW ? 1.f : 0.f;
    S[1] = g.n_stock / 52.f;
    S[2] = g.n_pile / 52.f;
    S[3] = g.turn / 200.f;
    S[4] = float(g.reshuffles);
    S[5] = g.drew_from_pile ? 1.f : 0.f;
    if (legal) {
        for (int a = 0; a < N_ACTIONS; ++a) legal[a] = false;
        if (g.phase == DRAW) legal[ACT_STOCK] = legal[ACT_PILE] = true;
        else if (g.phase == DISCARD)
            for (u64 h = g.hand[p]; h; h &= h - 1) legal[lowest(h)] = true;
    }
}

static std::vector<int> card_list(u64 m) {
    std::vector<int> v;
    for (; m; m &= m - 1) v.push_back(lowest(m));
    return v;
}

static py::dict state_dict(const Game& g) {
    py::dict d;
    d["hands"] = py::make_tuple(card_list(g.hand[0]), card_list(g.hand[1]));
    d["draw_pile"] = std::vector<int>(g.stock, g.stock + g.n_stock);
    d["discard_pile"] = std::vector<int>(g.pile, g.pile + g.n_pile);
    d["to_move"] = g.to_move;
    d["phase"] = std::string(g.phase == DRAW ? "draw" : g.phase == DISCARD ? "discard" : "over");
    d["turn"] = g.turn;
    d["reshuffles"] = g.reshuffles;
    d["drawn"] = g.drawn;
    d["drew_from_pile"] = g.drew_from_pile;
    d["held_known"] = py::make_tuple(card_list(g.held_known[0]), card_list(g.held_known[1]));
    d["discarded_by"] = py::make_tuple(card_list(g.discarded_by[0]), card_list(g.discarded_by[1]));
    d["taken_by"] = py::make_tuple(card_list(g.taken_by[0]), card_list(g.taken_by[1]));
    d["recycled"] = card_list(g.recycled);
    d["last_recycled"] = card_list(g.last_recycled);
    d["winner"] = g.winner;
    d["points"] = g.points;
    d["declare_card"] = g.declare_card;
    d["capped"] = g.capped;
    d["cap_scores"] = py::make_tuple(g.cap_score[0], g.cap_score[1]);
    return d;
}

// --------------------------------------------------------------------------- //
// Imagining the hidden cards
//
// The player to move can't see the opponent's hand or the order of the draw
// pile. To look ahead, the search deals those hidden cards out at random in a
// way that fits everything the player has seen, many times over ("worlds").
//   * cards the opponent took from the pile and still holds stay in their hand;
//   * after a reshuffle the draw pile can only hold cards from that reshuffle,
//     so any other unseen card must be in the opponent's hand;
//   * the rest of the opponent's hand is picked from the remaining unseen cards,
//     favouring cards the network thinks they hold (`weight`, one per card);
//   * whatever is left becomes the draw pile, in random order.
// --------------------------------------------------------------------------- //

static inline double uniform01(u64& rng) { return (splitmix(rng) >> 11) * (1.0 / 9007199254740992.0); }

static bool sample_world(const Game& g, const float* weight, u64& rng, Game& out) {
    int p = g.to_move, o = 1 - p;
    u64 unseen = ALL_CARDS & ~(g.hand[p] | g.pile_mask | g.held_known[o]);
    u64 can_be_stock = g.reshuffles ? (unseen & g.last_recycled) : unseen;
    u64 opp = g.held_known[o] | (unseen & ~can_be_stock);
    int need = popcount(g.hand[o]) - popcount(opp);
    int m = 0;
    int cand[N_CARDS];
    double w[N_CARDS];
    for (u64 c = can_be_stock; c; c &= c - 1) {
        cand[m] = lowest(c);
        w[m] = std::max(weight ? double(weight[cand[m]]) : 1.0, 1e-3);
        ++m;
    }
    if (need < 0 || need > m || m - need != g.n_stock) return false;
    for (int k = 0; k < need; ++k) {
        double total = 0;
        for (int j = 0; j < m; ++j) total += w[j];
        double r = uniform01(rng) * total;
        int j = 0;
        while (j < m - 1 && r >= w[j]) r -= w[j++];
        opp |= bit(cand[j]);
        cand[j] = cand[m - 1];
        w[j] = w[m - 1];
        --m;
    }
    out = g;
    out.hand[o] = opp;
    for (int i = m - 1; i > 0; --i) std::swap(cand[i], cand[below(rng, i + 1)]);
    for (int i = 0; i < m; ++i) out.stock[i] = uint8_t(cand[i]);
    out.n_stock = m;
    out.rng = splitmix(rng);  // future reshuffles in this world differ too
    return true;
}

// --------------------------------------------------------------------------- //
// Search (looking ahead)
//
// For each game, the search imagines `worlds` versions of the hidden cards and
// grows a small game tree in each one (Monte Carlo tree search, as in the Othello
// project). At every position in a tree the network is asked two things: which
// moves look good (priors) and how the hand will end (value). Each simulation
// walks down the tree picking moves that look good but haven't been tried much,
// asks the network about the new position it reaches, and passes the answer back
// up. The answer is the number of visits each move got at the top, added up over
// all worlds; the training uses that as the better move choice to learn from.
// --------------------------------------------------------------------------- //

static constexpr int MAX_LEGAL = 14;

struct Node {
    Game g;
    int n = 0;                         // legal actions at this position
    uint8_t act[MAX_LEGAL];
    float prior[MAX_LEGAL], N[MAX_LEGAL], W[MAX_LEGAL];
    int child[MAX_LEGAL];
    bool expanded = false, terminal = false;
};

static int legal_actions(const Game& g, uint8_t* out) {
    if (g.phase == DRAW) {
        out[0] = ACT_STOCK;
        out[1] = ACT_PILE;
        return 2;
    }
    int k = 0;
    for (u64 h = g.hand[g.to_move]; h; h &= h - 1) out[k++] = uint8_t(lowest(h));
    return k;
}

// Result of a finished hand for player p, scaled into [-1, 1] (T9.2 / VALUE_SCALE).
static float final_value(const Game& g, int p) { return hand_reward(g, p) / VALUE_SCALE; }

static void new_node(Node& nd, const Game& g) {
    nd.g = g;
    nd.n = g.phase == OVER ? 0 : legal_actions(g, nd.act);
    for (int k = 0; k < nd.n; ++k) {
        nd.prior[k] = nd.N[k] = nd.W[k] = 0.f;
        nd.child[k] = -1;
    }
    nd.expanded = false;
    nd.terminal = g.phase == OVER;
}

// Priors from a network row over all 54 actions, renormalised over the legal ones.
static void set_priors(Node& nd, const float* row) {
    float total = 0.f;
    for (int k = 0; k < nd.n; ++k) total += (nd.prior[k] = std::max(row[nd.act[k]], 0.f));
    for (int k = 0; k < nd.n; ++k) nd.prior[k] = total > 0.f ? nd.prior[k] / total : 1.f / nd.n;
    nd.expanded = true;
}

static int pick(const Node& nd, float c_puct) {
    float total = 0.f;
    for (int k = 0; k < nd.n; ++k) total += nd.N[k];
    float root = std::sqrt(total + 1.f);
    int best = 0;
    float best_score = -1e30f;
    for (int k = 0; k < nd.n; ++k) {
        float q = nd.N[k] > 0.f ? nd.W[k] / nd.N[k] : 0.f;
        float score = q + c_puct * nd.prior[k] * root / (1.f + nd.N[k]);
        if (score > best_score) {
            best_score = score;
            best = k;
        }
    }
    return best;
}

// --------------------------------------------------------------------------- //
// Env: many games side by side
// --------------------------------------------------------------------------- //

static u64 mix_seed(u64 seed, u64 a, u64 b) {
    u64 s = seed ^ (a * 0xD1B54A32D192ED03ULL) ^ (b * 0x8CB92BA72F3D8DD7ULL);
    splitmix(s);
    return s;
}

class Env {
  public:
    Env(int n, u64 seed, int turn_cap) : games_(n), hands_played_(n, 0), seed_(seed), turn_cap_(turn_cap) {
        if (n <= 0) throw std::invalid_argument("n must be positive");
        for (int i = 0; i < n; ++i) reset(i, -1, -1);
    }

    int size() const { return int(games_.size()); }

    // New hand in game i. first = 0/1 picks who starts, -1 picks at random (R2.4).
    // seed = -1 derives the shuffle from the env seed, the game index and how many
    // hands this game has played, so a run is reproducible.
    void reset(int i, int first, long long seed) {
        check(i);
        u64 s = seed >= 0 ? u64(seed) : mix_seed(seed_, u64(i), hands_played_[i]++);
        if (first < 0) {
            u64 t = s ^ 0x5DEECE66DULL;
            first = below(t, 2);
        }
        reset_game(games_[i], s, first);
    }

    void reset_all() {
        for (int i = 0; i < size(); ++i) reset(i, -1, -1);
    }

    // Sets game i to an exact position at the start of `to_move`'s turn (before
    // the draw). Piles are listed bottom to top. For tests: every card must
    // appear exactly once, hands hold 13, and the discard pile is not empty.
    void load(int i, const std::vector<int>& hand0, const std::vector<int>& hand1,
              const std::vector<int>& draw_pile, const std::vector<int>& discard_pile, int to_move) {
        check(i);
        if (hand0.size() != HAND || hand1.size() != HAND) throw std::invalid_argument("hands must hold 13 cards");
        if (discard_pile.empty()) throw std::invalid_argument("discard pile must not be empty");
        if (to_move != 0 && to_move != 1) throw std::invalid_argument("to_move must be 0 or 1");
        u64 seen = 0;
        for (const auto* part : {&hand0, &hand1, &draw_pile, &discard_pile})
            for (int c : *part) {
                if (c < 0 || c >= N_CARDS || (seen & bit(c))) throw std::invalid_argument("each card must appear exactly once");
                seen |= bit(c);
            }
        if (seen != ALL_CARDS) throw std::invalid_argument("all 52 cards must be placed");
        Game& g = games_[i];
        g = Game{};
        for (int c : hand0) g.hand[0] |= bit(c);
        for (int c : hand1) g.hand[1] |= bit(c);
        for (int c : draw_pile) g.stock[g.n_stock++] = uint8_t(c);
        for (int c : discard_pile) push_pile(g, c);
        g.to_move = to_move;
        g.phase = DRAW;
        g.drawn = -1;
        g.winner = -1;
        g.declare_card = -1;
        g.rng = mix_seed(seed_, u64(i), 0x10AD);
    }

    // One action per game (ignored for games that are over). Returns the reward
    // for each seat [n, 2] (T9.2: winner +points, loser -points, cap 0) and
    // whether each game ended on this step.
    py::tuple step(py::array_t<int, py::array::c_style | py::array::forcecast> actions) {
        auto a = actions.unchecked<1>();
        if (a.shape(0) != size()) throw std::invalid_argument("need one action per game");
        for (int i = 0; i < size(); ++i) {
            const Game& g = games_[i];
            if (g.phase == OVER) continue;
            std::string err = legal_error(g, a(i));
            if (!err.empty()) throw std::invalid_argument("game " + std::to_string(i) + ": " + err);
        }
        py::array_t<float> rewards({size(), 2});
        py::array_t<bool> done(size());
        auto r = rewards.mutable_unchecked<2>();
        auto d = done.mutable_unchecked<1>();
        {
            py::gil_scoped_release nogil;
            for (int i = 0; i < size(); ++i) {
                Game& g = games_[i];
                r(i, 0) = r(i, 1) = 0.f;
                d(i) = false;
                if (g.phase == OVER) continue;
                if (apply_action(g, a(i), turn_cap_)) {
                    d(i) = true;
                    for (int p = 0; p < 2; ++p) r(i, p) = hand_reward(g, p);
                }
            }
        }
        return py::make_tuple(rewards, done);
    }

    py::array_t<bool> legal_mask() const {
        py::array_t<bool> out({size(), N_ACTIONS});
        auto m = out.mutable_unchecked<2>();
        for (int i = 0; i < size(); ++i) {
            const Game& g = games_[i];
            for (int a = 0; a < N_ACTIONS; ++a) m(i, a) = false;
            if (g.phase == DRAW) {
                m(i, ACT_STOCK) = true;
                m(i, ACT_PILE) = true;
            } else if (g.phase == DISCARD) {
                for (u64 h = g.hand[g.to_move]; h; h &= h - 1) m(i, lowest(h)) = true;
            }
        }
        return out;
    }

    // What the player to move can see (R8): 10 card planes [n, 10, 52] and 6
    // numbers [n, 6]. See fill_obs for what each one holds.
    py::tuple observe() const {
        py::array_t<float> planes({size(), N_PLANES, N_CARDS});
        py::array_t<float> scalars({size(), N_SCALARS});
        float* P = planes.mutable_data();
        float* S = scalars.mutable_data();
        for (int i = 0; i < size(); ++i) fill_obs(games_[i], P + i * N_PLANES * N_CARDS, S + i * N_SCALARS, nullptr);
        return py::make_tuple(planes, scalars);
    }

    // For each game, the opponent's real hand (from the player to move's side)
    // as [n, 52] 0/1. This is hidden information: it is only used as the answer
    // when training the network to guess the opponent's hand.
    py::array_t<float> opponent_hands() const {
        py::array_t<float> out({size(), N_CARDS});
        auto o = out.mutable_unchecked<2>();
        for (int i = 0; i < size(); ++i) {
            u64 h = games_[i].hand[1 - games_[i].to_move];
            for (int c = 0; c < N_CARDS; ++c) o(i, c) = (h >> c & 1) ? 1.f : 0.f;
        }
        return out;
    }

    py::array_t<int> greedy_actions() const {
        py::array_t<int> out(size());
        auto o = out.mutable_unchecked<1>();
        py::gil_scoped_release nogil;
        for (int i = 0; i < size(); ++i) o(i) = games_[i].phase == OVER ? -1 : greedy_action(games_[i]);
        return out;
    }

    py::array_t<int> random_actions(u64 seed) const {
        py::array_t<int> out(size());
        auto o = out.mutable_unchecked<1>();
        u64 rng = seed;
        for (int i = 0; i < size(); ++i) o(i) = games_[i].phase == OVER ? -1 : random_action(games_[i], rng);
        return out;
    }

    py::array_t<int> to_move() const { return field([](const Game& g) { return g.to_move; }); }
    py::array_t<int> phase() const { return field([](const Game& g) { return g.phase; }); }
    py::array_t<bool> done() const {
        py::array_t<bool> out(size());
        auto o = out.mutable_unchecked<1>();
        for (int i = 0; i < size(); ++i) o(i) = games_[i].phase == OVER;
        return out;
    }

    // Everything about game i, including hidden cards. For tests, debugging and
    // the web server (which only shows the human what they may see).
    py::dict state(int i) const {
        check(i);
        return state_dict(games_[i]);
    }

    // One imagined version of game i's hidden cards, seen from the player to
    // move (for tests). weights: 52 numbers, how likely each card is in the
    // opponent's hand; empty = all equally likely.
    py::dict sample_world_state(int i, std::vector<float> weights, u64 seed) {
        check(i);
        if (!weights.empty() && weights.size() != N_CARDS) throw std::invalid_argument("weights must have 52 entries");
        Game w;
        u64 rng = seed;
        if (!sample_world(games_[i], weights.empty() ? nullptr : weights.data(), rng, w))
            throw std::runtime_error("hidden cards don't fit what the player has seen");
        return state_dict(w);
    }

    // Looks ahead for the player to move in every game where active[i] is true.
    //   evaluate(planes [L,10,52], scalars [L,6], legal [L,54])
    //       -> (priors [L,54], values [L], hand_guess [L,52])
    //   is the network, called once per simulation step for all games at once.
    //   Values are from the point of view of the player to move, in [-1, 1].
    // Returns (visits [n,54] summed over worlds, the network's value for each
    // game's current position [n], its guess of the opponent's hand [n,52]).
    py::tuple search(py::function evaluate, py::array_t<bool, py::array::c_style | py::array::forcecast> active,
                     int worlds, int sims, float c_puct, float dir_alpha, float dir_eps, u64 seed) {
        auto act_in = active.unchecked<1>();
        if (act_in.shape(0) != size()) throw std::invalid_argument("need one active flag per game");
        if (worlds < 1 || sims < 0) throw std::invalid_argument("worlds must be >= 1 and sims >= 0");
        py::array_t<float> visits({size(), N_ACTIONS});
        py::array_t<float> root_values(size());
        py::array_t<float> hand_guess({size(), N_CARDS});
        float* V = visits.mutable_data();
        float* RV = root_values.mutable_data();
        float* HG = hand_guess.mutable_data();
        std::fill(V, V + size_t(size()) * N_ACTIONS, 0.f);
        std::fill(RV, RV + size(), 0.f);
        std::fill(HG, HG + size_t(size()) * N_CARDS, 0.f);

        std::vector<int> roots;
        for (int i = 0; i < size(); ++i)
            if (act_in(i) && games_[i].phase != OVER) roots.push_back(i);
        if (roots.empty()) return py::make_tuple(visits, root_values, hand_guess);
        int R = int(roots.size());

        // Ask the network about all the positions in `games` at once.
        auto ask = [&](const std::vector<const Game*>& games) {
            int L = int(games.size());
            py::array_t<float> planes({L, N_PLANES, N_CARDS});
            py::array_t<float> scalars({L, N_SCALARS});
            py::array_t<bool> legal({L, N_ACTIONS});
            float* P = planes.mutable_data();
            float* S = scalars.mutable_data();
            bool* G = legal.mutable_data();
            for (int j = 0; j < L; ++j)
                fill_obs(*games[j], P + size_t(j) * N_PLANES * N_CARDS, S + size_t(j) * N_SCALARS, G + size_t(j) * N_ACTIONS);
            py::tuple out = evaluate(planes, scalars, legal);
            auto pri = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(out[0]);
            auto val = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(out[1]);
            auto hnd = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(out[2]);
            if (!pri || !val || !hnd || pri.size() != py::ssize_t(L) * N_ACTIONS || val.size() != py::ssize_t(L) || hnd.size() != py::ssize_t(L) * N_CARDS)
                throw std::runtime_error("evaluate must return priors [L,54], values [L], hand guesses [L,52]");
            return std::make_tuple(pri, val, hnd);
        };

        // 1. The real positions: priors, value and the guess of the opponent's hand.
        std::vector<const Game*> root_games;
        for (int i : roots) root_games.push_back(&games_[i]);
        auto [rpri, rval, rhnd] = ask(root_games);
        std::vector<float> root_row(size_t(R) * N_ACTIONS);
        std::copy(rpri.data(), rpri.data() + root_row.size(), root_row.begin());
        for (int r = 0; r < R; ++r) {
            RV[roots[r]] = rval.data()[r];
            std::copy(rhnd.data() + size_t(r) * N_CARDS, rhnd.data() + size_t(r + 1) * N_CARDS, HG + size_t(roots[r]) * N_CARDS);
        }

        // Exploration noise on the real positions' priors (self-play only).
        std::mt19937_64 gen(seed);
        if (dir_eps > 0.f) {
            std::gamma_distribution<double> gamma(dir_alpha, 1.0);
            for (int r = 0; r < R; ++r) {
                uint8_t acts[MAX_LEGAL];
                int n = legal_actions(games_[roots[r]], acts);
                double noise[MAX_LEGAL], total = 0;
                for (int k = 0; k < n; ++k) total += (noise[k] = gamma(gen));
                float* row = root_row.data() + size_t(r) * N_ACTIONS;
                float legal_total = 0.f;
                for (int k = 0; k < n; ++k) legal_total += std::max(row[acts[k]], 0.f);
                for (int k = 0; k < n; ++k) {
                    float prior = legal_total > 0.f ? std::max(row[acts[k]], 0.f) / legal_total : 1.f / n;
                    row[acts[k]] = (1.f - dir_eps) * prior + dir_eps * float(noise[k] / (total > 0 ? total : 1.0));
                }
            }
        }

        // 2. One tree per imagined world.
        int T = R * worlds;
        std::vector<std::vector<Node>> trees(T);
        std::vector<int> owner(T);   // player the tree is searching for
        u64 rng = seed ^ 0xA5A5A5A5DEADBEEFULL;
        for (int r = 0; r < R; ++r)
            for (int w = 0; w < worlds; ++w) {
                int t = r * worlds + w;
                Game world;
                if (!sample_world(games_[roots[r]], HG + size_t(roots[r]) * N_CARDS, rng, world))
                    throw std::runtime_error("hidden cards don't fit what the player has seen");
                trees[t].reserve(size_t(sims) + 1);
                trees[t].emplace_back();
                new_node(trees[t][0], world);
                set_priors(trees[t][0], root_row.data() + size_t(r) * N_ACTIONS);
                owner[t] = world.to_move;
            }

        // 3. Simulations, all trees in step so the network sees one big batch.
        std::vector<std::vector<std::pair<int, int>>> paths(T);
        auto backup = [&](int t, float v_owner) {
            for (auto [node, k] : paths[t]) {
                Node& nd = trees[t][node];
                nd.N[k] += 1.f;
                nd.W[k] += nd.g.to_move == owner[t] ? v_owner : -v_owner;
            }
        };
        std::vector<int> leaf_tree, leaf_node;
        std::vector<const Game*> leaf_games;
        for (int s = 0; s < sims; ++s) {
            leaf_tree.clear();
            leaf_node.clear();
            leaf_games.clear();
            for (int t = 0; t < T; ++t) {
                auto& tree = trees[t];
                paths[t].clear();
                int node = 0;
                while (true) {
                    int k = pick(tree[node], c_puct);
                    paths[t].push_back({node, k});
                    int next = tree[node].child[k];
                    if (next < 0) {
                        Game g2 = tree[node].g;
                        apply_action(g2, tree[node].act[k], turn_cap_);
                        next = int(tree.size());
                        tree.emplace_back();                 // may move nodes: use indices only
                        new_node(tree[next], g2);
                        tree[node].child[k] = next;
                        if (tree[next].terminal) backup(t, final_value(g2, owner[t]));
                        else {
                            leaf_tree.push_back(t);
                            leaf_node.push_back(next);
                        }
                        break;
                    }
                    node = next;
                    if (tree[node].terminal) {
                        backup(t, final_value(tree[node].g, owner[t]));
                        break;
                    }
                }
            }
            if (leaf_tree.empty()) continue;
            for (size_t j = 0; j < leaf_tree.size(); ++j) leaf_games.push_back(&trees[leaf_tree[j]][leaf_node[j]].g);
            auto [lpri, lval, lhnd] = ask(leaf_games);
            (void)lhnd;
            for (size_t j = 0; j < leaf_tree.size(); ++j) {
                int t = leaf_tree[j];
                Node& nd = trees[t][leaf_node[j]];
                set_priors(nd, lpri.data() + j * N_ACTIONS);
                float v = lval.data()[j];                     // for the leaf's player to move
                backup(t, nd.g.to_move == owner[t] ? v : -v);
            }
        }

        // 4. Add up the visits at the top of every world's tree.
        for (int r = 0; r < R; ++r)
            for (int w = 0; w < worlds; ++w) {
                const Node& root = trees[r * worlds + w][0];
                for (int k = 0; k < root.n; ++k) V[size_t(roots[r]) * N_ACTIONS + root.act[k]] += root.N[k];
            }
        return py::make_tuple(visits, root_values, hand_guess);
    }

    // Rule checks for every game; returns a list of "game i: problem" strings.
    std::vector<std::string> validate() const {
        std::vector<std::string> errs;
        for (int i = 0; i < size(); ++i) {
            std::string e = validate_game(games_[i]);
            if (!e.empty()) errs.push_back("game " + std::to_string(i) + ": " + e);
        }
        return errs;
    }

    // Plays full hands bot vs bot inside C++ (no Python per move) for benchmarks
    // and stress tests. bots[s] is "greedy" or "random" for seat s. Returns
    // (hands finished, total turns, seat-0 wins, seat-1 wins, capped, seat-0 net
    // points including capped hands (T9.1), rule violations found).
    py::tuple play(int hands, const std::string& bot0, const std::string& bot1, bool check_rules) {
        bool greedy[2] = {bot0 == "greedy", bot1 == "greedy"};
        for (const std::string* b : {&bot0, &bot1})
            if (*b != "greedy" && *b != "random") throw std::invalid_argument("bot must be greedy or random");
        long long finished = 0, turns = 0, wins[2] = {0, 0}, capped = 0, net0 = 0, violations = 0;
        {
            py::gil_scoped_release nogil;
            u64 rng = mix_seed(seed_, 0xB07, 0);
            int started = 0;
            for (int i = 0; i < size() && started < hands; ++i, ++started) reset(i, started & 1, -1);
            std::vector<bool> active(size(), false);
            for (int i = 0; i < std::min(size(), hands); ++i) active[i] = true;
            int live = std::min(size(), hands);
            while (live > 0) {
                for (int i = 0; i < size(); ++i) {
                    if (!active[i]) continue;
                    Game& g = games_[i];
                    int a = greedy[g.to_move] ? greedy_action(g) : random_action(g, rng);
                    bool ended = apply_action(g, a, turn_cap_);
                    if (check_rules && !validate_game(g).empty()) ++violations;
                    if (!ended) continue;
                    ++finished;
                    turns += g.turn;
                    if (g.capped) ++capped;
                    else wins[g.winner]++;
                    net0 += (long long)hand_reward(g, 0);
                    if (started < hands) reset(i, (started++) & 1, -1);
                    else { active[i] = false; --live; }
                }
            }
        }
        return py::make_tuple(finished, turns, wins[0], wins[1], capped, net0, violations);
    }

  private:
    template <class F>
    py::array_t<int> field(F f) const {
        py::array_t<int> out(size());
        auto o = out.mutable_unchecked<1>();
        for (int i = 0; i < size(); ++i) o(i) = f(games_[i]);
        return out;
    }
    void check(int i) const {
        if (i < 0 || i >= size()) throw std::out_of_range("game index out of range");
    }

    std::vector<Game> games_;
    std::vector<u64> hands_played_;
    u64 seed_;
    int turn_cap_;
};

// --------------------------------------------------------------------------- //
// Python module
// --------------------------------------------------------------------------- //

static u64 mask_arg(u64 m, int want) {
    if (m & ~ALL_CARDS) throw std::invalid_argument("card mask has bits above 51");
    if (popcount(m) != want) throw std::invalid_argument("expected " + std::to_string(want) + " cards, got " + std::to_string(popcount(m)));
    return m;
}

PYBIND11_MODULE(rummy_native, m) {
    m.doc() = "Rummy engine: scoring, game, observations, bots (see RULES.md)";
    m.attr("ACT_STOCK") = ACT_STOCK;
    m.attr("ACT_PILE") = ACT_PILE;
    m.attr("N_ACTIONS") = N_ACTIONS;
    m.attr("N_PLANES") = N_PLANES;
    m.attr("N_SCALARS") = N_SCALARS;
    m.attr("VALUE_SCALE") = VALUE_SCALE;

    m.def("score_hand", [](u64 mask) {
        int s = score13(mask_arg(mask, 13));
        return py::make_tuple(s == 0, s);
    }, "13-card mask -> (is_win, min_score)");
    m.def("best_discard", [](u64 mask) {
        int d, s;
        best_discard(mask_arg(mask, 14), d, s);
        return py::make_tuple(s == 0, s, d);
    }, "14-card mask -> (is_win, min_score, card_to_throw)");
    m.def("discard_scores", [](u64 mask) {
        int cards[14], scores[14];
        discard_scores(mask_arg(mask, 14), cards, scores);
        std::vector<std::pair<int, int>> out;
        for (int i = 0; i < 14; ++i) out.push_back({cards[i], scores[i]});
        return out;
    }, "14-card mask -> [(card, score of the 13 kept if that card is thrown)]");

    py::class_<Env>(m, "Env")
        .def(py::init<int, u64, int>(), py::arg("n"), py::arg("seed") = 0, py::arg("turn_cap") = 200)
        .def("__len__", &Env::size)
        .def("reset", &Env::reset, py::arg("i"), py::arg("first") = -1, py::arg("seed") = -1)
        .def("reset_all", &Env::reset_all)
        .def("load", &Env::load, py::arg("i"), py::arg("hand0"), py::arg("hand1"), py::arg("draw_pile"),
             py::arg("discard_pile"), py::arg("to_move") = 0)
        .def("step", &Env::step)
        .def("legal_mask", &Env::legal_mask)
        .def("observe", &Env::observe)
        .def("greedy_actions", &Env::greedy_actions)
        .def("random_actions", &Env::random_actions, py::arg("seed"))
        .def("to_move", &Env::to_move)
        .def("phase", &Env::phase)
        .def("done", &Env::done)
        .def("state", &Env::state)
        .def("opponent_hands", &Env::opponent_hands)
        .def("sample_world", &Env::sample_world_state, py::arg("i"), py::arg("weights") = std::vector<float>{}, py::arg("seed") = 0)
        .def("search", &Env::search, py::arg("evaluate"), py::arg("active"), py::arg("worlds") = 8, py::arg("sims") = 32,
             py::arg("c_puct") = 1.5f, py::arg("dir_alpha") = 0.5f, py::arg("dir_eps") = 0.f, py::arg("seed") = 0)
        .def("validate", &Env::validate)
        .def("play", &Env::play, py::arg("hands"), py::arg("bot0"), py::arg("bot1"), py::arg("check_rules") = false);
}
