"""
visualize_network.py — Draw a trained network's wiring.

Nodes = neurons, edges = weights. This shows how the layers actually connect
*after training*: every input is wired to every hidden neuron, and every hidden
neuron to every output (a "fully connected" / dense network).

    teal edge   = positive weight  (this input EXCITES that neuron)
    coral edge  = negative weight  (this input INHIBITS that neuron)
    thickness   = |weight|         (how strongly the neuron cares about it)

That's the same "how much this neuron cares about that input" idea from the
forward pass — just made visible. Matplotlib only, no extra dependencies.

Usage:
    python visualize_network.py --model champion_rl_v4.npz --out assets/network.png
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BG = "#0f0f1e"
NODE_IN = "#5dcaa5"
NODE_HID = "#7f77dd"
NODE_OUT = "#e0b85d"
POS = "#5dcaa5"     # positive weight
NEG = "#dd778c"     # negative weight
TXT = "#c8c8d2"

INPUT_LABELS = ["ball x", "ball y", "ball vx", "ball vy", "paddle y"]
X_GAP = 4.0
HEIGHT = 6.0


def load(path: str):
    """Return (layer_sizes, [weight matrices]) for either network type."""
    data = np.load(path)
    if "kind" in data and str(data["kind"]) == "dqn":
        from train_rl import QNetwork
        net = QNetwork.load(path)
        return net.layer_sizes, [net.W1, net.W2]
    from network import NeuralNetwork
    net = NeuralNetwork.load(path)
    return net.layer_sizes, net.weights


def layer_positions(sizes: list[int]):
    """Place each layer in a column, all spread over the same height."""
    pos = []
    for li, n in enumerate(sizes):
        x = li * X_GAP
        ys = [0.0] if n == 1 else list(np.linspace(HEIGHT / 2, -HEIGHT / 2, n))
        pos.append([(x, y) for y in ys])
    return pos


def output_labels(n: int):
    if n == 3:
        return ["Q: down (-1)", "Q: stay (0)", "Q: up (+1)"]
    if n == 1:
        return ["paddle move"]
    return [f"out {i}" for i in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sizes, weights = load(args.model)
    pos = layer_positions(sizes)

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ── Edges (weights), drawn behind the nodes ──
    for li, W in enumerate(weights):          # W: (out_size, in_size)
        scale = np.abs(W).max() or 1.0
        for i in range(W.shape[0]):           # node in next layer
            for j in range(W.shape[1]):       # node in this layer
                w = W[i, j]
                x0, y0 = pos[li][j]
                x1, y1 = pos[li + 1][i]
                a = abs(w) / scale
                ax.plot([x0, x1], [y0, y1],
                        color=POS if w >= 0 else NEG,
                        linewidth=0.2 + 2.6 * a,
                        alpha=0.07 + 0.5 * a, zorder=1)

    # ── Nodes ──
    node_colors = [NODE_IN] + [NODE_HID] * (len(sizes) - 2) + [NODE_OUT]
    for li, layer in enumerate(pos):
        xs = [x for x, _ in layer]
        ys = [y for _, y in layer]
        ax.scatter(xs, ys, s=380, color=node_colors[li],
                   edgecolors="white", linewidths=0.8, zorder=3)

    # ── Labels ──
    for j, (x, y) in enumerate(pos[0]):
        name = INPUT_LABELS[j] if j < len(INPUT_LABELS) else f"in {j}"
        ax.text(x - 0.45, y, name, ha="right", va="center", color=TXT, fontsize=11)
    for i, (x, y) in enumerate(pos[-1]):
        ax.text(x + 0.45, y, output_labels(sizes[-1])[i],
                ha="left", va="center", color=TXT, fontsize=11)

    titles = ["INPUT", "HIDDEN (ReLU)", "OUTPUT"]
    for li, layer in enumerate(pos):
        label = titles[li] if li < 2 else titles[-1]
        ax.text(layer[0][0], HEIGHT / 2 + 0.9, f"{label}\n{sizes[li]} neurons",
                ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")

    ax.set_title(f"How the network connects — {args.model}  {sizes}",
                 color="white", fontsize=14, pad=26)

    legend = [
        Line2D([0], [0], color=POS, lw=3, label="positive weight (excites)"),
        Line2D([0], [0], color=NEG, lw=3, label="negative weight (inhibits)"),
        Line2D([0], [0], color="#888", lw=0.6, label="thin = weak  ·  thick = strong"),
    ]
    leg = ax.legend(handles=legend, loc="lower center", ncol=3,
                    frameon=False, bbox_to_anchor=(0.5, -0.08))
    for t in leg.get_texts():
        t.set_color(TXT)

    ax.set_xlim(-2.2, (len(sizes) - 1) * X_GAP + 2.4)
    ax.set_ylim(-HEIGHT / 2 - 1.6, HEIGHT / 2 + 2.2)
    ax.axis("off")
    plt.savefig(args.out, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"wrote {args.out}  (layers {sizes})")


if __name__ == "__main__":
    main()
