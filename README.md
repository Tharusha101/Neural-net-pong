# NeuroPong

Teaching a neural network to play Pong — first with **neuroevolution** (a genetic algorithm), then with **reinforcement learning** (DQN) — built from scratch in NumPy so you can see exactly how a neural network works.

This is a learning project. Every file is heavily commented. The goal isn't just a working AI — it's to *understand* neural nets by building one with no black-box libraries.

## See it play

![Defender rally](assets/defender_rally.gif)

The network learns to **rally** from scratch (above — an 8-hit exchange, never
missing). For the full analysis — how it learns, why scoring against the bot is
hard, and the reward-shaping journey — see **[REPORT.md](REPORT.md)**.

## What you'll learn

- **What a neural network actually is** — `network.py` builds one from scratch: weights, biases, activations, forward pass. No PyTorch hiding the math.
- **How training = searching for good weights** — evolution and RL are just two different search strategies.
- **Neuroevolution** — the intuitive Darwin approach: population → fitness → selection → mutation → repeat.
- **Reinforcement learning** — the modern gradient approach (coming in phase 2), so you can compare.

## Files

| File | What it teaches |
|------|----------------|
| `network.py` | A neural network from scratch in NumPy — read this first |
| `pong.py` | The game environment: state in, action out (the ML interface) |
| `train_evolution.py` | Neuroevolution — train by survival of the fittest |
| `watch.py` | Render a trained network playing (pygame) |
| `train_rl.py` | Reinforcement learning with DQN — gradients/backprop from scratch (Phase 2) |

## Quick Start

```bash
pip install numpy pygame

# Phase 1 — train with evolution (watch fitness improve each generation)
python train_evolution.py --generations 50 --population 100

# Phase 2 — train the same-sized network with deep Q-learning (DQN)
python train_rl.py --episodes 2000

# Watch a trained champion play (evolution or RL)
python watch.py --model champion_evolution.npz
python watch.py --model champion_rl.npz
```

You'll see output like:
```
Gen   1 | best:    0.0 | avg:    0.0
Gen  15 | best:   22.0 | avg:    1.9    ← it's learning!
Gen  50 | best:   45.0 | avg:   12.3
```

Fitness = ball hits + 10×points scored. Watch it climb.

## The Learning Path

### Phase 1 — Neuroevolution (start here)

1. **Read `network.py` top to bottom.** This is a complete neural network in ~150 lines. Understand: inputs → weights → activation → output. That's the whole thing.
2. **Read `pong.py`'s `get_state()`** — see exactly what 5 numbers the network "sees."
3. **Run `train_evolution.py`** and watch the fitness climb generation by generation.
4. **Run `watch.py`** to see your evolved network play.
5. **Experiment** (this is where learning happens):
   - Change `LAYER_SIZES` to `[5, 16, 1]` — does a bigger brain learn faster?
   - Change the `fitness()` function — what if you only reward scoring, not hits?
   - Change `mutation_rate` — too high and it never settles, too low and it never improves.

### Phase 2 — Reinforcement Learning (the comparison)

`train_rl.py` trains the *same-sized* network with Deep Q-Learning. Unlike
evolution, a single network improves *itself* with **gradients** (backprop, by
hand in NumPy). Read it top to bottom — it's where you learn Q-values, the
Bellman equation, experience replay, epsilon-greedy exploration, target
networks, and reward shaping. Then compare: which learns faster? Which plays
better?

The included setup learns to **rally** (it reliably tracks and returns the ball)
but rarely scores against the opponent yet — a perfect place to experiment: try
`--episodes 5000`, a bigger hidden layer (`[5, 16, 3]`), or turn off the reward
shaping (`SHAPE_COEF = 0.0`) to feel how much it was helping.

## Ideas to extend

- Make the opponent smarter and see if the AI keeps up
- Add ball spin / acceleration and see if the network adapts
- Two AIs playing each other, both evolving (co-evolution)
- Plot the fitness curve over generations with matplotlib
- Visualize what the hidden neurons "light up" on
