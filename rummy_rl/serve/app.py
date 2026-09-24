"""Local website: play rummy against the greedy bot, and check any hand.

    python serve/app.py        # -> http://127.0.0.1:8001

The C++ engine (native/rummy_native.cpp) runs the game and the bot. This file
keeps the match score, turns engine state into what the human may see (never
the bot's hidden cards), and writes the move log in plain sentences.
engine/hand.py arranges hands into melds for display.

Pages:
  GET  /          play against the greedy bot (frontend/play.html)
  GET  /checker   hand checker (frontend/checker.html)

API:
  POST /api/game/new             start a match, returns the view
  GET  /api/game/{id}            current view
  POST /api/game/{id}/move       {action}: the human's move (52 draw pile, 53 discard pile, 0-51 throw)
  POST /api/game/{id}/bot        the bot makes one move (a draw or a throw)
  POST /api/game/{id}/next       deal the next hand after one ends
  POST /api/evaluate             {cards: [...]}: hand checker
"""

import os
import random
import sys
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))

from hand import HAND_SIZE, VALUE, arrange_hand, best_discard, card_str, parse_card, to_mask
from native import rummy_native as rn

FRONTEND_DIR = os.path.join(_HERE, "frontend")
_NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate"}
TARGET = 201           # R7.3
HUMAN, BOT = 0, 1
SYMBOL = dict(zip("SHDC", "♠♥♦♣"))

app = FastAPI(title="Rummy")


def pretty(c):
    t = card_str(c)
    return t[:-1] + SYMBOL[t[-1]]


def arrangement(cards):
    """Melds and scoring cards for a 13-card hand, ready for JSON."""
    _, score, melds, leftovers = arrange_hand(cards)
    return {
        "score": score,
        "melds": [{"kind": "set" if len({c % 13 for c in m}) == 1 else "run",
                   "cards": [card_str(c) for c in m]} for m in melds],
        "leftovers": [{"card": card_str(c), "value": VALUE[c]} for c in leftovers],
    }


# --------------------------------------------------------------------------- #
# Match
# --------------------------------------------------------------------------- #

class Match:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.env = rn.Env(1, seed=random.getrandbits(62), turn_cap=0)
        self.scores = [0, 0]
        self.hand_no = 0
        self.first = random.randint(0, 1)   # R2.4
        self.lock = threading.Lock()
        self.new_hand()

    def new_hand(self):
        first = self.first ^ (self.hand_no % 2)   # alternate who starts
        self.hand_no += 1
        self.env.reset(0, first=first)
        self.log = ["You start this hand." if first == HUMAN else "The bot starts this hand."]
        self.result = None

    @property
    def state(self):
        return self.env.state(0)

    def match_winner(self):
        if self.scores[HUMAN] >= TARGET:
            return "bot"
        if self.scores[BOT] >= TARGET:
            return "you"
        return None

    def act(self, player, action):
        s = self.state
        if s["phase"] == "over":
            raise HTTPException(409, "This hand is over. Deal the next hand.")
        if s["to_move"] != player:
            raise HTTPException(409, "It's not your turn." if player == HUMAN else "It's your turn, not the bot's.")
        legal = self.env.legal_mask()[0]
        if not (0 <= action < len(legal)) or not legal[action]:
            raise HTTPException(400, "That move isn't allowed right now.")
        top = s["discard_pile"][-1] if s["discard_pile"] else None
        reshuffles = s["reshuffles"]
        self.env.step([action])
        after = self.state
        who = "You" if player == HUMAN else "The bot"

        if action == rn.ACT_STOCK:
            if after["reshuffles"] > reshuffles:
                self.log.append("The draw pile ran out, so the discard pile (all but its top card) "
                                "was shuffled into a new draw pile.")
            self.log.append(f"You drew {pretty(after['drawn'])} from the draw pile."
                            if player == HUMAN else "The bot drew a card from the draw pile.")
        elif action == rn.ACT_PILE:
            self.log.append(f"{who} took {pretty(top)} from the discard pile.")
        else:
            self.log.append(f"{who} threw {pretty(action)}.")

        if after["phase"] == "over":
            self.finish_hand(after)

    def finish_hand(self, s):
        winner, points = s["winner"], s["points"]
        loser = 1 - winner
        self.scores[loser] += points          # R7.2
        if winner == HUMAN:
            self.log.append(f"You declared and threw {pretty(s['declare_card'])}. "
                            f"The bot is caught with {points} points.")
        else:
            self.log.append(f"The bot declared and threw {pretty(s['declare_card'])}. "
                            f"You are caught with {points} points.")
        self.result = {
            "winner": "you" if winner == HUMAN else "bot",
            "points": points,
            "declare_card": card_str(s["declare_card"]),
            "you": arrangement(s["hands"][HUMAN]),
            "bot": arrangement(s["hands"][BOT]),
        }

    def view(self):
        s = self.state
        mine = s["hands"][HUMAN]
        if len(mine) == HAND_SIZE + 1:
            _, _, extra = best_discard(mine)
            arr = arrangement([c for c in mine if c != extra])
            arr["extra"] = card_str(extra)
        else:
            arr = arrangement(mine)
            arr["extra"] = None
        drawn = s["drawn"] if s["phase"] == "discard" and s["to_move"] == HUMAN else -1
        return {
            "id": self.id,
            "hand_no": self.hand_no,
            "target": TARGET,
            "scores": {"you": self.scores[HUMAN], "bot": self.scores[BOT]},
            "match_winner": self.match_winner(),
            "phase": s["phase"],
            "turn": "you" if s["to_move"] == HUMAN else "bot",
            "you": {"cards": [card_str(c) for c in mine],
                    "drawn": card_str(drawn) if drawn >= 0 else None,
                    "arrangement": arr},
            "bot": {"count": len(s["hands"][BOT]),
                    "known": [card_str(c) for c in s["held_known"][BOT]]},
            "draw_pile": len(s["draw_pile"]),
            "discard_top": card_str(s["discard_pile"][-1]) if s["discard_pile"] else None,
            "discard_count": len(s["discard_pile"]),
            "log": self.log[-8:],
            "result": self.result,
        }


MATCHES = {}


def get_match(match_id):
    m = MATCHES.get(match_id)
    if m is None:
        raise HTTPException(404, "That game has ended. Start a new match.")
    return m


class MoveRequest(BaseModel):
    action: int


@app.post("/api/game/new")
def new_game():
    m = Match()
    MATCHES[m.id] = m
    return m.view()


@app.get("/api/game/{match_id}")
def game_view(match_id: str):
    return get_match(match_id).view()


@app.post("/api/game/{match_id}/move")
def game_move(match_id: str, req: MoveRequest):
    m = get_match(match_id)
    with m.lock:
        m.act(HUMAN, req.action)
        return m.view()


@app.post("/api/game/{match_id}/bot")
def game_bot(match_id: str):
    m = get_match(match_id)
    with m.lock:
        s = m.state
        if s["phase"] == "over" or s["to_move"] != BOT:
            return m.view()
        m.act(BOT, int(m.env.greedy_actions()[0]))
        return m.view()


@app.post("/api/game/{match_id}/next")
def game_next(match_id: str):
    m = get_match(match_id)
    with m.lock:
        if m.state["phase"] != "over":
            raise HTTPException(409, "This hand isn't over yet.")
        if m.match_winner():
            raise HTTPException(409, "The match is over. Start a new match.")
        m.new_hand()
        return m.view()


# --------------------------------------------------------------------------- #
# Hand checker
# --------------------------------------------------------------------------- #

class EvaluateRequest(BaseModel):
    cards: list[str]


@app.get("/")
def play_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "play.html"), headers=_NO_CACHE)


@app.get("/checker")
def checker_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "checker.html"), headers=_NO_CACHE)


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    try:
        cards = [parse_card(t) for t in req.cards]
        to_mask(cards)  # rejects duplicates
    except ValueError as e:
        raise HTTPException(400, str(e))
    if len(cards) not in (HAND_SIZE, HAND_SIZE + 1):
        raise HTTPException(400, f"need {HAND_SIZE} or {HAND_SIZE + 1} cards, got {len(cards)}")

    discard = None
    if len(cards) == HAND_SIZE + 1:
        _, _, discard = best_discard(cards)
        cards = [c for c in cards if c != discard]
    result = arrangement(cards)
    result.update({
        "win": result["score"] == 0,
        "total": sum(VALUE[c] for c in cards),
        "discard": card_str(discard) if discard is not None else None,
    })
    return result


if __name__ == "__main__":
    import uvicorn

    print("Rummy -> http://127.0.0.1:8001  (hand checker at /checker)")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
