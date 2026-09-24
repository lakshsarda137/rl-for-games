# Rummy: rules specification

The single source of truth for the game engine, the training environment, and the website. Two player, one deck, no jokers. Rule IDs (R1, R2, ...) are stable so code and tests can reference them.

## 1. Cards

- **R1.1** One standard 52-card deck. No jokers, no wild cards, no duplicate cards.
- **R1.2** Ranks: `A 2 3 4 5 6 7 8 9 10 J Q K`. Suits: spades, hearts, diamonds, clubs. Suits have no order or value.

## 2. Deal

- **R2.1** Shuffle the deck. Deal 13 cards to each player.
- **R2.2** Turn the next card face up. It starts the **discard pile**.
- **R2.3** The remaining 25 cards, face down, are the **draw pile**.
- **R2.4** The first player of the first hand is chosen at random. After that, the first player alternates every hand.

## 3. Turn

A turn is always one draw followed by one discard.

- **R3.1 Draw.** Take either the top card of the draw pile or the top card of the discard pile. The player now holds 14 cards.
- **R3.2 Discard.** Place one of the 14 cards face up on the discard pile. The player now holds 13 cards.
- **R3.3** The card taken from the discard pile may be discarded again on the same turn.
- **R3.4 Declare.** Instead of a plain discard, the player may discard and declare, if the 13 cards left in hand form a valid hand (R5). Declaring ends the hand immediately.
  The engine declares automatically: as soon as a draw makes a valid declare possible, it throws the spare card and declares, because declaring is always the best move.
- **R3.5** A player can only declare on their own turn, after drawing. Being dealt a valid hand does not end the game; the player must still draw and discard.
- **R3.6** The first player may take the face-up card from R2.2 on their first turn.

## 4. Empty draw pile

- **R4.1** If the draw pile is empty when a player needs to draw from it, take every card in the discard pile **except the top card**, shuffle them, and turn them face down. That is the new draw pile.
- **R4.2** The top card stays on the discard pile.
- **R4.3** There is no limit on the number of reshuffles in a real game.

## 5. Melds and a valid hand

- **R5.1 Run.** 3 or more cards of the same suit in consecutive rank, e.g. `5♥ 6♥ 7♥`. No upper length limit.
- **R5.2 Set.** 3 or 4 cards of the same rank, all different suits, e.g. `9♠ 9♦ 9♣`. (A single deck has only 4 of each rank.)
- **R5.3 Ace.** Ace can be low (`A 2 3`) or high (`Q K A`). A run cannot wrap around: `K A 2` is not valid.
- **R5.4 Pure run.** With no jokers, every run is pure. The term is kept because R5.5 depends on it.
- **R5.5 Valid hand.** All 13 cards are split into melds with no card left over, and **at least one meld is a run of 4 or more cards**. The other melds can be any mix of runs (3+) and sets (3–4).

Examples of valid shapes (the first number is the 4+ run): `4+3+3+3`, `4+3+6`, `4+4+5`, `5+4+4`, `7+3+3`, `10+3`, a single 13-card run.

Not valid:
- Every meld has 3 cards (no run of 4+).
- The only 4-card groups are sets.
- Any card left unmelded.

- **R5.6** The engine only offers "declare" when the hand is valid, so there is no wrong-declaration penalty.

## 6. Scoring a hand

- **R6.1** The player who declares scores **0**.
- **R6.2** The other player scores the points in their 13-card hand, according to R6.3 and R6.4. Points are bad.
- **R6.3 Card values.** `A 10 J Q K` = 10 each. `2`–`9` = face value.
- **R6.4 Arrangement.**
  - If the loser's hand contains a run of 4 or more, their cards are arranged into melds to give the **lowest possible score**. Cards in melds score 0, and every leftover card scores its value.
  - If the loser's hand contains no run of 4 or more, **every card scores**, even cards that would form sets or 3-card runs.
  - The arrangement is computed automatically and optimally. The loser does not choose it.
- **R6.5** There is no maximum score for a hand.
- **R6.6** The loser cannot add cards to the winner's melds.

Example: the loser holds `6♥ 7♥ 8♥ 9♥ K♠ K♥ K♦ 2♠ 5♣ 9♦ Q♥ 3♠ 7♣`.
- `6♥–9♥` is a run of 4, so melding is allowed.
- `K♠ K♥ K♦` is a set.
- Leftovers: `2 + 5 + 9 + 10 + 3 + 7 = 36` points.

## 7. Match

- **R7.1** A match is a series of hands. Each player's score starts at 0.
- **R7.2** After each hand, the loser adds their hand score (R6.2) to their match score. The winner adds nothing.
- **R7.3** The first player whose match score reaches **201 or more** loses the match.
- **R7.4** Only one player scores per hand, so a match cannot end in a tie.

## 8. Information

What each player knows. The training environment's observation must follow this exactly.

- **R8.1 Private:** your own hand.
- **R8.2 Public:**
  - The top card of the discard pile.
  - Every card ever discarded, and by whom.
  - Every card taken from the discard pile, and by whom.
  - How many cards are in the draw pile.
  - How many reshuffles have happened, and which cards went into each reshuffle.
  - Both match scores.
- **R8.3 Hidden:**
  - The opponent's hand, apart from the cards you saw them take from the discard pile and haven't yet seen them discard.
  - The order of the draw pile.
- **R8.4** Players have perfect memory. Anything that was ever public stays known.

## 9. Training-only settings

These do not exist in the real game. They only keep self-play finite. The website never uses them.

- **T9.1 Turn cap.** If a hand reaches **200 turns** (both players combined) without a declaration, it ends as a **draw**: neither player scores.
- **T9.2 Hand reward.** For per-hand training, the reward is the point difference.
  - The winner gets `+` the loser's points.
  - The loser gets `−` their own points.
  - A capped draw gives 0 to both.
- **T9.3 Match reward (optional stage).** +1 for winning the match, −1 for losing it.
