"""
train_rl.py — Teach the network to play Pong using DEEP Q-LEARNING (DQN).

This is Phase 2 of NeuroPong, the counterpart to train_evolution.py.

In Phase 1 we found good weights by *survival of the fittest*: no calculus,
just mutate-and-select across a whole population. No single network ever
learned anything — the population improved because winners reproduced.

Here we do the opposite, and the modern thing: ONE network that edits its own
weights, a tiny step at a time, by GRADIENT DESCENT. It feels its mistakes and
nudges every weight in the exact direction that makes the mistake smaller.

Everything is from scratch in NumPy — no PyTorch. The new ideas, all of which
are implemented below and explained at their definition:

  1. Q-VALUES. Instead of one "move" number, the network estimates, for EACH
     action (down / stay / up), "how much total future reward do I expect if I
     take this action now and play well afterwards?" Pick the biggest Q-value.

  2. THE BELLMAN EQUATION (the training target). We never know the true
     Q-value, so we bootstrap from our own estimate one step into the future:
         Q(s, a)   should equal   r  +  gamma * max_a' Q(s', a')
     "the value of this move = the reward I just got + the best I can do next
     (shrunk by gamma, because future reward is worth a little less)."

  3. BACKPROPAGATION + GRADIENT DESCENT. We measure how wrong the prediction
     was, then push every weight a small step downhill. This is the "learning
     gradients" the README promised — done by hand so you can see the calculus.

  4. EXPERIENCE REPLAY. We store past (state, action, reward, next_state)
     transitions in a buffer and train on RANDOM batches of them. Reusing old
     experience is far more data-efficient, and random sampling breaks the
     correlation between consecutive frames (which destabilizes training).

  5. EPSILON-GREEDY EXPLORATION. At first the network knows nothing, so it must
     try random moves to discover what works (EXPLORE). As it learns it trusts
     its Q-values more (EXPLOIT). epsilon = probability of a random move; it
     starts at 1.0 (all random) and decays toward 0.05.

  6. TARGET NETWORK. The Bellman target depends on the network's own estimate —
     a moving goalpost. We freeze a COPY of the network for computing targets
     and refresh it only occasionally. This stabilizes learning a lot.

Usage:
    python train_rl.py
    python train_rl.py --episodes 2000 --seed 0
    python watch.py --model champion_rl.npz      # then watch it play
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import deque

import numpy as np

from pong import PongGame, HEIGHT

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Network shape ────────────────────────────────────────────────────────────
# 5 inputs (the game state, same as evolution) → 8 hidden → 3 Q-values.
# The 3 outputs line up with the 3 actions below. This is the ONLY structural
# difference from the evolution network ([5, 8, 1]): we need one value PER
# action, not a single steering number.
LAYER_SIZES = [5, 8, 3]
ACTIONS = [-1, 0, 1]  # output index 0 → down, 1 → stay, 2 → up

# ── Hyperparameters (the knobs of RL — every one is worth experimenting with) ─
GAMMA = 0.95          # discount: how much future reward is worth vs. now
LR = 0.0005           # learning rate: how big a step each weight takes downhill
BATCH_SIZE = 64       # how many remembered transitions we learn from at once
BUFFER_SIZE = 20000   # how many past transitions we keep in memory
TARGET_UPDATE = 500   # refresh the frozen target network every N learning steps
EPS_START = 1.0       # start fully random (pure exploration)
EPS_END = 0.05        # end mostly greedy (5% random moves, a little exploration)
EPS_DECAY_FRAC = 0.7  # reach EPS_END after this fraction of all episodes
# Reward shaping (POTENTIAL-BASED): the raw env reward (+1 hit / -1 miss / +2
# score) is sparse — a fresh network flails for ages before it ever touches the
# ball, and with no signal in between it tends to collapse to a useless constant
# move (paddle stuck to a wall). So we add a dense hint built from a "potential"
#     Phi(s) = -(distance from paddle to ball)
# and shape the reward as  F = gamma * Phi(s') - Phi(s), which pays out every
# step that CLOSES the gap to the ball. This potential-based form is special: it
# provably leaves the OPTIMAL policy unchanged, so we can turn it up to learn
# faster without secretly teaching the wrong goal. Set SHAPE_COEF = 0.0 to train
# on the pure sparse reward and watch how much harder it is. (Compare fitness()
# in evolution: the agent becomes good at exactly what you reward.)
# Defense shaping is ANNEALED over training: strong early so it learns to rally
# fast, then decayed toward near-zero so the agent stops being pulled to dead-
# center the ball and is free to angle shots for offense (schedule in train()).
SHAPE_START = 2.0
SHAPE_END = 0.25
# Offense shaping — the "defender problem": rewarding only "don't miss" yields a
# perfect wall that never scores. A first attempt (reward steep returns) failed
# because it FOUGHT the defense reward: centering the ball on the paddle (safe,
# no angle) is exactly what "don't miss" wants, so the agent never angled.
# Fix: reward the OUTCOME, not the technique — while our shot travels toward the
# opponent, reward the opponent being far from the ball (a shot the 70%-speed
# bot can't reach). This pulls toward scoring without dictating risky edge-hits.
OPP_COEF = 2.0        # reward, per step the ball heads right, for displacing the bot
SCORE_BONUS = 5.0     # jackpot (on top of the env's +2) for actually scoring
# Pick the champion by how much it SCORES in greedy eval (selecting on hits just
# rewards stubborn defense). Evaluate every EVAL_EVERY episodes over EVAL_GAMES.
EVAL_EVERY = 250
EVAL_GAMES = 12
LOG_EVERY = 50        # print progress every N episodes


class QNetwork:
    """A 2-layer Q-network: state (5) → hidden (8, ReLU) → Q-values (3, linear).

    Same from-scratch spirit as network.py, but with two new powers:
      - it outputs one Q-value per action (linear output, not tanh)
      - it can run BACKWARD: given how wrong it was, it computes how to adjust
        every weight (backpropagation) and takes a gradient-descent step.
    """

    def __init__(self, layer_sizes: list[int]) -> None:
        self.layer_sizes = list(layer_sizes)
        in_size, hidden, out_size = layer_sizes

        # He initialization (same as network.py) keeps early signals sane.
        self.W1 = np.random.randn(hidden, in_size) * np.sqrt(2.0 / in_size)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(out_size, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(out_size)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, tuple]:
        """Run states through the network. Returns (Q_values, cache).

        X can be one state (shape (5,)) or a batch (shape (B, 5)). The cache
        holds the intermediate values we'll need to run backward.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))  # (B, 5)
        Z1 = X @ self.W1.T + self.b1                        # (B, 8) pre-activation
        H = np.maximum(0, Z1)                               # (B, 8) ReLU
        Q = H @ self.W2.T + self.b2                         # (B, 3) LINEAR output
        return Q, (X, Z1, H)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Q-values for a single state, as a flat (3,) vector."""
        Q, _ = self.forward(state)
        return Q[0]

    # ── Choosing an action ───────────────────────────────────────────────────
    def act(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy: usually pick the best action, sometimes explore.

        Returns an ACTION INDEX (0/1/2), used during training.
        """
        if random.random() < epsilon:
            return random.randrange(len(ACTIONS))   # explore: random move
        return int(np.argmax(self.q_values(state)))  # exploit: best Q-value

    def decide(self, state: np.ndarray) -> int:
        """Greedy action as a game action (-1/0/+1).

        This matches network.NeuralNetwork.decide(), so pong.play_episode() and
        watch.py can drive a trained Q-network with zero special-casing.
        """
        return ACTIONS[int(np.argmax(self.q_values(state)))]

    # ── Backward pass: the actual learning ───────────────────────────────────
    def train_step(self, batch: list, target: "QNetwork") -> float:
        """Learn from one batch of remembered transitions. Returns the loss.

        Each transition is (state, action_index, reward, next_state, done).
        """
        states = np.array([t[0] for t in batch], dtype=np.float64)       # (B, 5)
        actions = np.array([t[1] for t in batch], dtype=np.int64)        # (B,)
        rewards = np.array([t[2] for t in batch], dtype=np.float64)      # (B,)
        next_states = np.array([t[3] for t in batch], dtype=np.float64)  # (B, 5)
        dones = np.array([t[4] for t in batch], dtype=np.float64)        # (B,)

        # 1) What the network currently predicts for the actions we took.
        Q, (X, Z1, H) = self.forward(states)            # (B, 3)

        # 2) The Bellman TARGET, using the frozen target network for stability.
        #    target = reward + gamma * best-next-Q   (but no future if the game
        #    ended this step, so we zero it out with (1 - done)).
        Q_next, _ = target.forward(next_states)         # (B, 3)
        targets = rewards + GAMMA * Q_next.max(axis=1) * (1.0 - dones)  # (B,)

        # 3) Error ONLY on the action we actually took — we have no target for
        #    the others, so their gradient is zero. dQ is dLoss/dQ for an MSE
        #    loss, averaged over the batch.
        idx = np.arange(len(batch))
        dQ = np.zeros_like(Q)                           # (B, 3)
        dQ[idx, actions] = (Q[idx, actions] - targets) / len(batch)

        # 4) Backpropagate that error through the two layers (the chain rule).
        dW2 = dQ.T @ H                  # (3, 8)
        db2 = dQ.sum(axis=0)           # (3,)
        dH = dQ @ self.W2              # (B, 8)  push error back through W2
        dZ1 = dH * (Z1 > 0)           # (B, 8)  ReLU derivative: 1 where active
        dW1 = dZ1.T @ X               # (8, 5)
        db1 = dZ1.sum(axis=0)         # (8,)

        # 5) Gradient descent: step every weight a little bit DOWNHILL.
        self.W1 -= LR * dW1
        self.b1 -= LR * db1
        self.W2 -= LR * dW2
        self.b2 -= LR * db2

        return float(0.5 * np.mean((Q[idx, actions] - targets) ** 2))

    # ── Bookkeeping ──────────────────────────────────────────────────────────
    def clone(self) -> "QNetwork":
        twin = QNetwork(self.layer_sizes)
        twin.copy_from(self)
        return twin

    def copy_from(self, other: "QNetwork") -> None:
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

    def save(self, path: str) -> None:
        # 'kind' lets watch.py tell a DQN file apart from an evolution file.
        np.savez(
            path,
            w0=self.W1, b0=self.b1, w1=self.W2, b1=self.b2,
            layer_sizes=np.array(self.layer_sizes), kind=np.array("dqn"),
        )

    @staticmethod
    def load(path: str) -> "QNetwork":
        data = np.load(path)
        net = QNetwork(data["layer_sizes"].tolist())
        net.W1, net.b1 = data["w0"], data["b0"]
        net.W2, net.b2 = data["w1"], data["b1"]
        return net


def evaluate_score(net: QNetwork, n_games: int) -> tuple[int, float]:
    """Play greedy games (no exploration) → (total points scored, avg hits).

    This is our model-selection yardstick. We keep the snapshot that SCORES the
    most, because selecting on hits just rewards stubborn defense.
    """
    game = PongGame()
    total_score = 0
    total_hits = 0
    for _ in range(n_games):
        stats = game.play_episode(net)   # play_episode uses net.decide() (greedy)
        total_score += stats["scored"]
        total_hits += stats["hits"]
    return total_score, total_hits / n_games


def train(episodes: int, seed: int | None = None,
          hidden: int = 8, out: str = "champion_rl.npz") -> QNetwork:
    """The DQN training loop: play, remember, replay, repeat."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    layer_sizes = [5, hidden, 3]          # 5 state inputs, `hidden` neurons, 3 Q-values
    env = PongGame(seed=seed)
    online = QNetwork(layer_sizes)        # the network we train every step
    target = online.clone()               # frozen copy for stable Bellman targets
    buffer: deque = deque(maxlen=BUFFER_SIZE)

    global_step = 0
    recent_hits: deque = deque(maxlen=LOG_EVERY)
    best_score = -1                       # champion chosen by greedy SCORE...
    best_hits = -1.0                      # ...with avg hits as the tiebreaker
    best_net = online.clone()

    logger.info(f"Starting DQN: {episodes} episodes, network {layer_sizes}")
    logger.info(f"gamma={GAMMA}  lr={LR}  batch={BATCH_SIZE}  "
                f"shaping={SHAPE_START}->{SHAPE_END}\n")

    for ep in range(episodes):
        state = env.reset()

        # Linearly decay exploration from EPS_START to EPS_END.
        frac = min(1.0, ep / (episodes * EPS_DECAY_FRAC))
        epsilon = EPS_START + frac * (EPS_END - EPS_START)
        # Anneal defense shaping from SHAPE_START down to SHAPE_END over the run:
        # rally first, then free the agent to play offense.
        shape_coef = SHAPE_START + (ep / episodes) * (SHAPE_END - SHAPE_START)

        done = False
        while not done:
            a_idx = online.act(state, epsilon)

            # Potential BEFORE the move: closer paddle-to-ball ⇒ higher potential.
            phi_prev = -abs(env.left_paddle_y - env.ball_y) / HEIGHT
            next_state, reward, done = env.step(ACTIONS[a_idx])
            # Potential AFTER the move (a terminal state has potential 0).
            phi_next = 0.0 if done else -abs(env.left_paddle_y - env.ball_y) / HEIGHT

            # Defense: dense, policy-preserving shaping that rewards closing the
            # gap (its strength `shape_coef` decays across training — see above).
            shaped = reward
            shaped += shape_coef * (GAMMA * phi_next - phi_prev)
            # Offense: while our return travels toward the opponent, reward the
            # opponent being out of position (far from the ball) — a shot it can't
            # chase. Pay the jackpot when we actually score.
            if env.ball_vx > 0:
                shaped += OPP_COEF * abs(env.ball_y - env.right_paddle_y) / HEIGHT
            if reward >= 2.0:
                shaped += SCORE_BONUS

            # Remember this transition, then learn from a random batch.
            buffer.append((state, a_idx, shaped, next_state, float(done)))
            state = next_state
            global_step += 1

            if len(buffer) >= BATCH_SIZE:
                online.train_step(random.sample(buffer, BATCH_SIZE), target)

            # Periodically copy the trained net into the frozen target net.
            if global_step % TARGET_UPDATE == 0:
                target.copy_from(online)

        recent_hits.append(env.hits)

        if (ep + 1) % LOG_EVERY == 0:
            avg = float(np.mean(recent_hits))
            logger.info(
                f"Ep {ep+1:4d} | eps {epsilon:.2f} | shape {shape_coef:.2f} | "
                f"avg hits (last {len(recent_hits)}): {avg:.2f}"
            )

        # Periodically measure GREEDY scoring; keep the best-scoring snapshot.
        if (ep + 1) % EVAL_EVERY == 0:
            score, eval_hits = evaluate_score(online, EVAL_GAMES)
            star = ""
            if (score, eval_hits) > (best_score, best_hits):
                best_score, best_hits = score, eval_hits
                best_net = online.clone()
                star = "  <- new best"
            logger.info(
                f"  [eval ep {ep+1}] scored {score}/{EVAL_GAMES} games, "
                f"avg hits {eval_hits:.1f}{star}"
            )

    logger.info(f"\nDone! Best greedy score: {best_score}/{EVAL_GAMES} games")
    best_net.save(out)
    logger.info(f"Saved best network to {out}")
    return best_net


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pong AI with Deep Q-Learning")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=8, help="hidden-layer size")
    parser.add_argument("--out", default="champion_rl.npz", help="where to save the champion")
    args = parser.parse_args()

    champion = train(episodes=args.episodes, seed=args.seed,
                     hidden=args.hidden, out=args.out)

    # Show how the champion does — reuses pong.play_episode via decide().
    logger.info("\nTesting champion over 10 games:")
    game = PongGame()
    for i in range(10):
        stats = game.play_episode(champion)
        logger.info(f"  Game {i+1}: {stats['hits']} hits, {stats['scored']} points scored")


if __name__ == "__main__":
    main()
