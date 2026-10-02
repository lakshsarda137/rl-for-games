"""How fast is self-play on this machine? python run/bench.py

Plays a few search steps for 512 games on each GPU (or the CPU) with the real
settings and prints moves per second. Run it on Kaggle to see what the T4s do.
"""

import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))
sys.path.insert(0, os.path.join(_HERE, "..", "az"))
sys.path.insert(0, _HERE)

from config import Config
from native import rummy_native as rn
from network import Evaluator, RummyNet, pick_device


def moves_per_second(device, cfg, games=512, steps=6):
    ev = Evaluator(RummyNet(cfg.blocks, cfg.channels), device)
    env = rn.Env(games, seed=3, turn_cap=cfg.turn_cap)
    for _ in range(30):                      # get into the middle of a hand
        env.step(env.greedy_actions())
    env.search(ev, ~env.done(), worlds=cfg.worlds, sims=cfg.sims, seed=0)   # warm up
    t = time.perf_counter()
    for s in range(steps):
        env.search(ev, ~env.done(), worlds=cfg.worlds, sims=cfg.sims, seed=s + 1)
        env.step(env.greedy_actions())
        for i in np.flatnonzero(env.done()):
            env.reset(int(i))
    return steps * games / (time.perf_counter() - t)


if __name__ == "__main__":
    cfg = Config()
    devices = [f"cuda:{k}" for k in range(torch.cuda.device_count())] or [pick_device()]
    print(f"Settings: {cfg.blocks} blocks x {cfg.channels} channels, {cfg.worlds} worlds x {cfg.sims} sims "
          f"= {cfg.worlds * cfg.sims} network calls per move", flush=True)
    for d in devices:
        name = torch.cuda.get_device_name(int(d.split(":")[1])) if d.startswith("cuda") else d
        print(f"{d} ({name}): {moves_per_second(d, cfg):,.0f} moves/s", flush=True)
    if len(devices) > 1:
        print(f"Training uses all {len(devices)} GPUs at once, so self-play runs at about the sum.")
