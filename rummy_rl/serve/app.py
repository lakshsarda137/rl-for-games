"""Local web page for the hand engine: click cards, see the verdict.

    python serve/app.py        # -> http://127.0.0.1:8001

A thin wrapper: engine/hand.py decides everything, this file only turns its
answer into JSON for serve/frontend/index.html.

Endpoints:
  GET  /               -> the page
  POST /api/evaluate   {cards: ["6H", "7H", ...]} with 13 or 14 cards ->
        {win, score, total, discard, melds: [{kind, cards}], leftovers: [{card, value}]}
        `discard` is set only for 14 cards (the engine's best discard); melds and
        leftovers then describe the 13 cards that are kept.
"""

import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))

from hand import HAND_SIZE, VALUE, arrange_hand, best_discard, card_str, parse_card, to_mask

FRONTEND_DIR = os.path.join(_HERE, "frontend")
_NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate"}

app = FastAPI(title="Rummy hand checker")


class EvaluateRequest(BaseModel):
    cards: list[str]


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"), headers=_NO_CACHE)


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
    win, score, melds, leftovers = arrange_hand(cards)
    return {
        "win": win,
        "score": score,
        "total": sum(VALUE[c] for c in cards),
        "discard": card_str(discard) if discard is not None else None,
        "melds": [{"kind": "set" if len({c % 13 for c in m}) == 1 else "run",
                   "cards": [card_str(c) for c in m]} for m in melds],
        "leftovers": [{"card": card_str(c), "value": VALUE[c]} for c in leftovers],
    }


if __name__ == "__main__":
    import uvicorn

    print("Rummy hand checker -> http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
