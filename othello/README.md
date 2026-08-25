# Othello AI

A computer program that taught itself to play Othello, plus a website where you can play against it.

**Play it here:** http://othello-alb-381409665.us-east-1.elb.amazonaws.com

No download or setup is needed. Open the link, press **New Game**, and click a highlighted square to place a disc.

## What is Othello?

Othello (also called Reversi) is a two player board game on an 8 by 8 grid. One player has black discs, the other has white. You place a disc so that one or more of your opponent's discs are trapped in a straight line between your new disc and one of your existing discs. Those trapped discs flip to your colour. When the board is full, or nobody can move, the player with more discs wins.

## How to use the website

- **Black** and **White** are the two players. By default you are Black and our AI is White. You move first.
- Each computer player has a **difficulty**: Easy, Medium, Hard, or Expert. Harder levels think longer before each move, so they play better but take a little more time.
- You can also set both players to computers and watch them play each other. The **Speed** slider controls the pause between computer moves. **Pause** and **Step** let you stop and move one turn at a time.
- The number next to each player is how many discs they currently have on the board.

## The computer players

| Name | What it is |
|---|---|
| **Our AI** | The program this project is about. It learned Othello by playing tens of thousands of games against itself, with no human strategy built in. See "How our AI learned" below. |
| **Edax** | A well known open source Othello program written by other people. It is one of the strongest Othello programs in the world. Even its Easy level is a serious challenge. We include it so you can see how our AI compares to the best. |
| **Minimax** | A classic way of programming board games. It looks a few moves ahead, imagines every possible reply, and picks the move that leads to the best position according to a fixed set of rules (for example, corners are good, giving your opponent many options is bad). |
| **Greedy** | Always plays the move that flips the most discs right now. Simple and easy to beat. |
| **Random** | Picks any legal move at random. |

## How our AI learned

Our AI follows the recipe made famous by AlphaZero, the program from DeepMind that taught itself chess and Go. Here is the idea in plain terms.

1. **It starts knowing nothing** except the rules. At first it plays random moves.
2. **It plays against itself.** Every game it plays is recorded: each board position, which move it chose, and who eventually won.
3. **It learns from those games.** A neural network (a program that improves at making predictions when shown many examples) is trained to answer two questions about any board position: "which moves look promising?" and "who is likely to win from here?"
4. **It looks ahead using those predictions.** Before each move it imagines many possible continuations, focusing on the moves the network thinks are promising. This lookahead is called Monte Carlo tree search. The lookahead produces better decisions than the network alone.
5. **Repeat.** The improved player plays more games against itself, the network learns from those, and so on. Each round it gets a little stronger.

The result is a player that was never told any Othello strategy, yet learned things like the value of corners on its own.

Training happened on a free Kaggle graphics card over several sessions. The main model has 10 layers of 128 units and went through about 125 rounds of self play and learning. Measured against a fixed opponent (Edax at a low level, 100 game matches) it went from winning about a quarter of games early on to winning roughly half.

## Running it on your own computer

You only need this if you want to change the code or train your own model. To just play, use the link at the top.

You need Python 3.11 and the packages `numpy`, `torch`, `fastapi`, and `uvicorn`.

```bash
cd othello
python run/play_cli.py --black human --white minimax:3   # play in the terminal
python serve/backend.py                                  # website at http://127.0.0.1:8000
python run_tests.py                                      # run the checks (about 10 seconds)
python run/train_loop.py --tiny                          # a tiny training run to see it work
```

The website needs a trained model file at `data/checkpoints/latest.pt` to offer "Our AI". Without one, you can still play Minimax, Greedy, and Random.

Optional extras:

- **Edax** is not included in this repository. `opponents/EDAX_SETUP.md` explains how to build it. The website works without it.
- **Faster training:** `python native/build.py` compiles a small piece of C++ code that makes self play much faster. Everything works without it, just slower.
- **Training on Kaggle** (free graphics card): see `run/KAGGLE.md`.
- **Developer view of the website:** add `?dev` to the address (for example `http://127.0.0.1:8000/?dev`). This shows every saved version of the model, an Arena for running many games at once to measure strength, a Models list, and a Tournament page where several computer players play each other.

## What is in this folder

```
othello/
  engine/       the rules of Othello (the rest of the code trusts this part)
  native/       optional C++ version of the rules and the lookahead, for speed
  opponents/    Minimax, Greedy, Random, and the Edax wrapper
  az/           the self learning player: neural network, lookahead, self play, training
  run/          scripts: train, play in the terminal, pull models from Kaggle
  serve/        the website (backend.py) and its web pages (frontend/)
  deploy/       files for running the website on a cloud server (Docker, AWS)
  tests/        automatic checks that everything still works
  data/         (not in git) trained models, game records, training logs
  third_party/  (not in git) the Edax program
```

## Words used in this project

- **Model** or **checkpoint**: a saved copy of the neural network at some point in training. `latest.pt` is the newest one.
- **Iteration** or **round**: one cycle of "play games against yourself, then learn from them".
- **Simulations**: how many possible continuations the AI imagines before choosing a move. The difficulty levels on the website set this for you.
- **Self play**: the AI playing games against a copy of itself to produce training data.
- **Neural network**: a program that learns to make predictions by being shown many examples, instead of being given rules by a person.
