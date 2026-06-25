"""
make_gif.py — Render a trained champion playing Pong into an animated GIF.

Pure Pillow (no pygame), so it runs headless. Mirrors watch.py's colors.

Usage:
    python make_gif.py --model champion_rl_big.npz --out assets/defender_rally.gif --mode rally
    python make_gif.py --model champion_rl_v4.npz  --out assets/attacker_score.gif --mode score
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from pong import PongGame, WIDTH, HEIGHT, PADDLE_HEIGHT

SCALE = 3
W, H = int(WIDTH * SCALE), int(HEIGHT * SCALE)
BG = (15, 15, 30)
MID = (45, 45, 68)
AI = (127, 119, 221)     # left paddle (the learner), purple
OPP = (120, 120, 130)    # right paddle (scripted bot), gray
BALL = (93, 202, 165)    # teal
TXT = (200, 200, 210)


def load_model(path: str):
    """Load either an evolution net (NeuralNetwork) or a DQN (QNetwork)."""
    data = np.load(path)
    if "kind" in data and str(data["kind"]) == "dqn":
        from train_rl import QNetwork
        return QNetwork.load(path)
    from network import NeuralNetwork
    return NeuralNetwork.load(path)


def draw_frame(game: PongGame, label: str = "") -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # Dashed center line
    for y in range(0, H, 18):
        d.rectangle([W // 2 - 1, y, W // 2 + 1, y + 9], fill=MID)
    ph = PADDLE_HEIGHT * SCALE
    # AI paddle (left)
    ly = game.left_paddle_y * SCALE
    d.rectangle([4, int(ly - ph / 2), 10, int(ly + ph / 2)], fill=AI)
    # Opponent paddle (right)
    ry = game.right_paddle_y * SCALE
    d.rectangle([W - 10, int(ry - ph / 2), W - 4, int(ry + ph / 2)], fill=OPP)
    # Ball
    bx, by, r = game.ball_x * SCALE, game.ball_y * SCALE, 3
    d.ellipse([int(bx - r), int(by - r), int(bx + r), int(by + r)], fill=BALL)
    if label:
        d.text((6, 4), label, fill=TXT)
    return img


def capture_episode(net, seed: int, frame_cap: int | None = None):
    """Play one greedy episode; return (frames, scored, hits)."""
    game = PongGame(seed=seed)
    state = game.reset()
    frames = [draw_frame(game, "rally: 0")]
    scored = False
    for _ in range(2000):
        state, reward, done = game.step(net.decide(state))
        frames.append(draw_frame(game, f"rally: {game.hits}"))
        if done:
            scored = reward >= 2.0   # +2 means the AI put it past the opponent
            break
        if frame_cap and len(frames) >= frame_cap:
            break
    return frames, scored, game.hits


def save_gif(frames, out: str, duration: int = 40) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    print(f"wrote {out}  ({len(frames)} frames)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["rally", "score"], default="rally")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--duration", type=int, default=40, help="ms per frame")
    p.add_argument("--max-seeds", type=int, default=400)
    args = p.parse_args()

    net = load_model(args.model)

    if args.mode == "score":
        # Search seeds for an episode the AI actually wins, keep its climax.
        for seed in range(args.max_seeds):
            frames, scored, hits = capture_episode(net, seed)
            if scored:
                print(f"scoring episode at seed {seed} ({hits} hits, {len(frames)} frames)")
                save_gif(frames[-args.frames:], args.out, args.duration)
                return
        print("no scoring episode found in the seed budget")
        raise SystemExit(1)

    # rally mode: show the longest rally among a handful of seeds, real-time.
    best, best_hits = None, -1
    for seed in range(24):
        frames, _, hits = capture_episode(net, seed, frame_cap=args.frames)
        if hits > best_hits:
            best, best_hits = frames, hits
    print(f"best rally: {best_hits} hits ({len(best)} frames)")
    save_gif(best, args.out, args.duration)


if __name__ == "__main__":
    main()
