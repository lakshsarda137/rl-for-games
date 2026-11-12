# Wordle: rules specification

The single source of truth for the game engine, the training setup and any website. Rule IDs (W1.1, W2.3, ...) are stable so code and tests can refer to them.

## 1. Words

- **W1.1** The secret word is chosen at random, with equal chance, from `data/answers.txt` (2,315 common five-letter words).
- **W1.2** A guess must be a word in `data/guesses.txt` (12,972 words). The first 2,315 lines of that file are the answer words in the same order as `answers.txt`, so an answer's line number is the same in both files.
- **W1.3** Guesses that are not in the list are rejected and do not use up a turn.

## 2. Colors (feedback)

After each guess, every letter of the guess gets one color.

- **W2.1 Green.** The letter is in the secret word at this exact position.
- **W2.2 Yellow.** The letter is in the secret word at a different position.
- **W2.3 Gray.** The letter is not in the secret word, or every copy of it is already marked by W2.4.
- **W2.4 Repeated letters.** Greens are marked first. Then, going from left to right, a letter gets yellow only while the secret still has an unmarked copy of it. Example: secret `those`, guess `geese` gives gray, gray, gray, green, green. The secret has one `e`, and the green `e` at the end uses it.
- **W2.5** A color pattern is stored as one number from 0 to 242: each position is a digit (gray 0, yellow 1, green 2), and the first letter is the lowest digit. All green is 242.

## 3. Solo game

- **W3.1** One player makes up to 6 guesses.
- **W3.2** The game ends when a guess is all green (solved) or after 6 guesses (failed).
- **W3.3** Score: the number of guesses used. A failed game scores 7. Lower is better.

## 4. Duel (two players)

- **W4.1** One secret word (W1.1) is shared by both players.
- **W4.2** The players take turns. Each turn is one guess.
- **W4.3** Each player sees only their own guesses and their own colors. A player does not see the other player's guesses or colors. Both players know how many turns have passed.
- **W4.4** The first player to make an all-green guess wins.
- **W4.5** If 12 guesses (6 per player) are made without a win, the game is a draw.
- **W4.6** The first player is chosen at random for the first game of a match. After that, the first player alternates every game.

## 5. Duel scoring

Faster wins earn more points, and a loss that came close costs fewer points.

- **W5.1 Winner.** The winner scores 7 minus the number of guesses the winner made. A win in 2 guesses scores +5. A win in 4 guesses scores +3.
- **W5.2 Loser.** The loser's penalty depends on how many answers still fit the loser's own colors when the game ends:
  - 1 answer left: −1
  - 2 to 10 answers left: −2
  - more than 10 answers left: −3
- **W5.3 Draw.** If nobody wins (W4.5), both players score 0.
- **W5.4** The AI's training reward is its own score from W5.1 to W5.3.

Why this duel is different from two solo games: a player does not know how close the other player is to the answer, only how many turns have passed. Each turn, a player must choose between a safe guess that rules out many words and a risky guess of a word that might be the answer. Later in the game, the other player is more likely to win on their next turn, so the risky guess becomes more worth it.
