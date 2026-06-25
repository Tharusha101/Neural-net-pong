"""
visualize_activation.py — Watch one decision flow through the network.

The wiring diagram (visualize_network.py) shows the *weights*. This shows the
*signal*: feed a real game state in, run the forward pass, and light up what
actually fires.

    input node brightness  = the value it received
    hidden node brightness = its ReLU activation (DARK = silenced to 0)
    edge brightness        = signal carried = (source activation x weight)
    highlighted output     = the action the network chose (argmax Q)

The key thing to watch: hidden neurons ReLU'd to zero go dark, and their
outgoing edges vanish — you can literally see which neurons the network is
(and isn't) using for this particular situation.

Usage:
    python visualize_activation.py --model champion_rl_v4.npz --out assets/activation.png
    python visualize_activation.py --model champion_rl_v4.npz --out assets/activation.gif --animate
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image

from pong import PongGame, PADDLE_HEIGHT, WIDTH, HEIGHT

BG = "#0f0f1e"
MID = "#2c2c44"
NODE_IN = "#5dcaa5"
NODE_HID = "#7f77dd"
NODE_OUT = "#e0b85d"
POS = "#5dcaa5"
NEG = "#dd778c"
TXT = "#c8c8d2"

INPUT_LABELS = ["ball x", "ball y", "ball vx", "ball vy", "paddle y"]
ACTION_NAME = {1: "up (+1)", -1: "down (-1)", 0: "stay (0)"}
X_GAP = 4.0
LAYER_H = 6.0


def load(path: str):
    """Return (net, layer_sizes, weights, biases, kind)."""
    data = np.load(path)
    if "kind" in data and str(data["kind"]) == "dqn":
        from train_rl import QNetwork
        net = QNetwork.load(path)
        return net, net.layer_sizes, [net.W1, net.W2], [net.b1, net.b2], "dqn"
    from network import NeuralNetwork
    net = NeuralNetwork.load(path)
    return net, net.layer_sizes, net.weights, net.biases, "evo"


def activations(weights, biases, x, kind):
    """Forward pass that keeps every layer's activations."""
    a = np.asarray(x, dtype=float)
    acts = [a]
    for li, (W, b) in enumerate(zip(weights, biases)):
        z = W @ a + b
        a = np.maximum(0, z) if li < len(weights) - 1 else (z if kind == "dqn" else np.tanh(z))
        acts.append(a)
    return acts


def layer_positions(sizes):
    pos = []
    for li, n in enumerate(sizes):
        x = li * X_GAP
        ys = [0.0] if n == 1 else list(np.linspace(LAYER_H / 2, -LAYER_H / 2, n))
        pos.append([(x, y) for y in ys])
    return pos


def scales_over(weights, acts_list):
    """Brightness normalizers, shared across frames so they're comparable."""
    hmax, s0, s1 = 1e-9, 1e-9, 1e-9
    for acts in acts_list:
        hmax = max(hmax, acts[1].max())
        s0 = max(s0, np.abs(acts[0][None, :] * weights[0]).max())
        s1 = max(s1, np.abs(acts[1][None, :] * weights[1]).max())
    return hmax, [s0, s1]


def draw_board(ax, f):
    ax.set_facecolor(BG)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.set_aspect("equal")
    for y in range(0, int(HEIGHT), 8):
        ax.add_patch(Rectangle((WIDTH / 2 - 0.4, y), 0.8, 4, color=MID))
    ax.add_patch(Rectangle((2, f["lpy"] - PADDLE_HEIGHT / 2), 3, PADDLE_HEIGHT, color=NODE_IN))
    ax.add_patch(Rectangle((WIDTH - 5, f["rpy"] - PADDLE_HEIGHT / 2), 3, PADDLE_HEIGHT, color="#8a8a99"))
    ax.add_patch(Circle((f["bx"], f["by"]), 2.6, color="#5dcaa5"))
    ax.set_title("what it sees", color=TXT, fontsize=10)
    ax.axis("off")


def draw_net(ax, sizes, weights, acts, q, scales):
    ax.set_facecolor(BG)
    pos = layer_positions(sizes)
    hmax, sig = scales

    # Edges — brightness = signal carried (source activation x weight)
    for li, W in enumerate(weights):
        src, s = acts[li], sig[li]
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                a = abs(src[j] * W[i, j]) / s
                if a < 0.03:
                    continue
                a = min(a, 1.0)
                x0, y0 = pos[li][j]
                x1, y1 = pos[li + 1][i]
                ax.plot([x0, x1], [y0, y1],
                        color=POS if src[j] * W[i, j] >= 0 else NEG,
                        linewidth=0.2 + 2.8 * a, alpha=0.05 + 0.7 * a, zorder=1)

    # Input nodes
    for j, (x, y) in enumerate(pos[0]):
        ax.scatter([x], [y], s=340, color=NODE_IN, edgecolors="white", linewidths=0.6, zorder=3)
        ax.text(x - 0.5, y, f"{INPUT_LABELS[j]}\n{acts[0][j]:+.2f}",
                ha="right", va="center", color=TXT, fontsize=9)

    # Hidden nodes — brightness = ReLU activation (dark = silenced)
    for j, (x, y) in enumerate(pos[1]):
        frac = min(acts[1][j] / hmax, 1.0)
        if frac > 0.25:
            ax.scatter([x], [y], s=380 + 1500 * frac, color=NODE_HID, alpha=0.16 * frac, zorder=2)
        ax.scatter([x], [y], s=340, color=NODE_HID, alpha=0.16 + 0.84 * frac,
                   edgecolors="white" if frac > 0.5 else "#3a3a55", linewidths=0.7, zorder=3)

    # Output nodes — highlight the chosen action
    labels = ["down (-1)", "stay (0)", "up (+1)"] if sizes[-1] == 3 else ["paddle move"]
    idx = int(np.argmax(q)) if sizes[-1] > 1 else 0
    for i, (x, y) in enumerate(pos[-1]):
        chosen = i == idx
        ax.scatter([x], [y], s=560 if chosen else 340, color=NODE_OUT,
                   edgecolors="white" if chosen else "#555", linewidths=2.2 if chosen else 0.7, zorder=4)
        ax.text(x + 0.5, y, f"{labels[i]}\nQ={q[i]:+.2f}", ha="left", va="center",
                color="white" if chosen else TXT, fontsize=9,
                fontweight="bold" if chosen else "normal")

    for li, name in enumerate(["INPUT", "HIDDEN (ReLU)", "OUTPUT"][:len(sizes)]):
        ax.text(pos[li][0][0], LAYER_H / 2 + 0.8, name, ha="center", va="bottom",
                color="white", fontsize=10, fontweight="bold")

    ax.set_xlim(-2.6, (len(sizes) - 1) * X_GAP + 3.0)
    ax.set_ylim(-LAYER_H / 2 - 1.2, LAYER_H / 2 + 1.5)
    ax.axis("off")


def play_frames(net, weights, biases, kind, n, seed):
    """Play n greedy frames, resetting on point-end so we always get n."""
    game = PongGame(seed=seed)
    state = game.reset()
    frames = []
    for _ in range(n):
        acts = activations(weights, biases, state, kind)
        action = net.decide(state)
        frames.append(dict(bx=game.ball_x, by=game.ball_y,
                           lpy=game.left_paddle_y, rpy=game.right_paddle_y,
                           acts=acts, q=np.atleast_1d(acts[-1]), action=action))
        state, _, done = game.step(action)
        if done:
            state = game.reset()
    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--animate", action="store_true")
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    net, sizes, weights, biases, kind = load(args.model)
    frames = play_frames(net, weights, biases, kind, max(args.frames, 120), args.seed)

    if args.animate:
        frames = frames[:args.frames]
        scales = scales_over(weights, [f["acts"] for f in frames])
        fig, (axb, axn) = plt.subplots(1, 2, figsize=(9, 4.5), dpi=72,
                                       gridspec_kw={"width_ratios": [1, 2.6]})
        fig.patch.set_facecolor(BG)
        images = []
        for f in frames:
            axb.clear()
            axn.clear()
            draw_board(axb, f)
            draw_net(axn, sizes, weights, f["acts"], f["q"], scales)
            fig.suptitle(f"one decision flowing through the net   ->   chose: {ACTION_NAME[f['action']]}",
                         color="white", fontsize=12)
            fig.canvas.draw()
            img = Image.frombytes("RGBA", fig.canvas.get_width_height(),
                                  fig.canvas.buffer_rgba()).convert("RGB")
            images.append(img)
        plt.close(fig)
        images[0].save(args.out, save_all=True, append_images=images[1:],
                       duration=90, loop=0, optimize=True)
        print(f"wrote {args.out}  ({len(images)} frames)")
        return

    # Static: among frames where the ball is approaching and it decides to move,
    # pick the most DECISIVE one (largest gap between the top two Q-values).
    cands = [fr for fr in frames if fr["action"] != 0 and fr["bx"] < 45]

    def margin(fr):
        s = np.sort(fr["q"])
        return s[-1] - s[-2]

    f = max(cands, key=margin) if cands else frames[len(frames) // 2]
    scales = scales_over(weights, [f["acts"]])
    fig, (axb, axn) = plt.subplots(1, 2, figsize=(11, 5.5),
                                   gridspec_kw={"width_ratios": [1, 2.6]})
    fig.patch.set_facecolor(BG)
    draw_board(axb, f)
    draw_net(axn, sizes, weights, f["acts"], f["q"], scales)
    n_live = int((f["acts"][1] > 1e-6).sum())
    fig.suptitle(f"one decision flowing through the net   ->   chose: {ACTION_NAME[f['action']]}"
                 f"      ({n_live}/{sizes[1]} hidden neurons firing)",
                 color="white", fontsize=13)
    plt.savefig(args.out, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"wrote {args.out}  ({n_live}/{sizes[1]} hidden neurons active)")


if __name__ == "__main__":
    main()
