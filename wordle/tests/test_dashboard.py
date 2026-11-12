"""Checks for the dashboard server (serve/app.py), without a browser."""

import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from fastapi.testclient import TestClient

from engine import words
from engine.game import Game
from serve import app as dash
from train import train_loop


def main():
    t0 = time.time()
    tmp = tempfile.mkdtemp()
    old_models = dash.MODELS_DIR
    dash.MODELS_DIR = os.path.join(tmp, "models")
    try:
        # A tiny trained model to load.
        run = os.path.join(tmp, "run")
        train_loop.main(["--tiny", "--no-wandb", "--out", run, "--iterations", "3", "--eval-every", "0"])
        c = TestClient(dash.app)
        assert c.get("/").status_code == 200

        # Upload a model through the page's upload route.
        with open(os.path.join(run, "latest.pt"), "rb") as f:
            r = c.post("/api/models/upload?name=latest.pt", content=f.read())
        assert r.status_code == 200, r.text
        model = r.json()["id"]
        bad = c.post("/api/models/upload?name=junk.pt", content=b"not a model")
        assert bad.status_code == 400
        ids = [m["id"] for m in c.get("/api/models").json()]
        assert "untrained" in ids and model in ids

        # Mode 1: play full games against the AI. The AI's guesses stay hidden until the end.
        v = c.post("/api/human/new", json={"model": model}).json()
        for game in range(4):
            assert v["your_rows"] == [] and not v["over"]
            assert v["ai_guess_count"] == (0 if v["you_first"] else 1)
            assert "ai_rows" not in v and "secret" not in v
            assert c.post(f"/api/human/{v['id']}/guess", json={"word": "zzzzz"}).status_code == 400
            assert c.post(f"/api/human/{v['id']}/guess", json={"word": "abc"}).status_code == 400
            for word in ["soare", "clint", "bumpy", "fight", "wreck", "vodka"]:
                v = c.post(f"/api/human/{v['id']}/guess", json={"word": word}).json()
                if v["over"]:
                    break
                assert "ai_rows" not in v
                assert v["ai_guess_count"] == len(v["your_rows"]) + (0 if v["you_first"] else 1)
            if not v["over"]:  # the AI may still need its last guess; it plays right after ours
                raise AssertionError("game should be over after 6 guesses each")
            assert v["secret"] in words.ANSWERS and v["winner"] in ("you", "ai", "nobody")
            assert all(r["options"] and r["left_before"] >= 1 for r in v["ai_rows"])
            assert v["games_played"] == game + 1
            first = v["you_first"]
            v = c.post(f"/api/human/{v['id']}/next").json()
            assert v["you_first"] != first  # W4.6

        # Mode 1: guessing the secret wins at once.
        v = c.post("/api/human/new", json={"model": model}).json()
        secret = dash.matches[v["id"]]["game"].secret
        v = c.post(f"/api/human/{v['id']}/guess", json={"word": secret}).json()
        if v["you_first"]:
            assert v["over"] and v["winner"] == "you" and v["points"]["you"] == 6
        else:
            assert v["over"] and v["winner"] in ("you", "ai")

        # You can choose the secret word; words outside the answer list are refused.
        v = c.post("/api/human/new", json={"model": model, "secret": "crane"}).json()
        assert v["chosen_word"] and dash.matches[v["id"]]["game"].secret == "crane"
        assert c.post("/api/human/new", json={"model": model, "secret": "zzzzz"}).status_code == 400
        v = c.post(f"/api/human/{v['id']}/next", json={"secret": "pupil"}).json()
        assert v["chosen_word"] and dash.matches[v["id"]]["game"].secret == "pupil"
        v = c.post(f"/api/human/{v['id']}/next", json={}).json()
        assert not v["chosen_word"]

        # The AI never sees the secret word (W4.3).
        net, _ = dash.load_model(model)
        # 1. Remove the secret from the game object: the AI can still choose, so it never reads it.
        g = Game(secret="crane", players=2)
        g.guess("slate")  # player 0 (the human) guesses
        del g.secret, g.secret_index
        dash.ai_turn(net, g)
        # 2. Same colors seen means same choice, whatever the secret is.
        firsts = set()
        for secret in ["crane", "pupil", "zesty", "gawky", "abbey", "those", "fjord", "vivid"]:
            g = Game(secret=secret, players=2)
            firsts.add(dash.ai_turn(net, g)[0])
        assert len(firsts) == 1, firsts
        # 3. The AI's own choice function has no way to receive the secret.
        import inspect
        assert list(inspect.signature(dash.ai_choose).parameters) == [
            "net", "my_candidates", "guesses_used", "went_first", "top_k"]

        # Mode 2: one game, then many.
        g = c.post("/api/watch/game", json={"model": model, "secret": "crane", "ai_first": False}).json()
        assert g["secret"] == "crane" and g["turns"][0]["player"] == "bot" and g["turns"][0]["word"] == "soare"
        assert g["turns"][-1]["colors"] == ["green"] * 5 or g["winner"] == "nobody"
        assert all("options" in t for t in g["turns"] if t["player"] == "ai")
        assert c.post("/api/watch/game", json={"model": model, "secret": "zzzzz"}).status_code == 400
        m = c.post("/api/watch/many", json={"model": model, "words": 30}).json()
        assert m["games"] == 60 and 0 <= m["ai"]["win_pct"] <= 100
        assert c.post("/api/watch/many", json={"model": "untrained", "words": 5}).status_code == 200
        assert c.post("/api/human/new", json={"model": "../../etc/passwd"}).status_code == 404
    finally:
        dash.MODELS_DIR = old_models
        shutil.rmtree(tmp)
    print(f"all dashboard checks passed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
