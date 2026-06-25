"""
train_evolution.py — Teach the network to play Pong using NEUROEVOLUTION.

This is the fun, intuitive way to train. There's no backpropagation, no
gradients, no calculus. The whole algorithm is basically Darwin:

    1. Make a POPULATION of random networks (they all play terribly)
    2. Let each one play Pong and measure how well it did (FITNESS)
    3. Keep the best ones (SELECTION)
    4. Make children from them via MUTATION (and crossover)
    5. Repeat for many GENERATIONS

Over time, the population gets better at Pong — not because any single
network learned, but because good networks survived and reproduced.

This teaches you what a neural net fundamentally IS: a set of weights that
maps inputs to outputs. Training = searching for good weights. Evolution is
just one search strategy (gradient descent is another — that's the RL version).

Usage:
    python train_evolution.py
    python train_evolution.py --generations 100 --population 150
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from network import NeuralNetwork
from pong import PongGame

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Network shape: 5 inputs (the game state) → 8 hidden neurons → 1 output (move)
LAYER_SIZES = [5, 8, 1]


def fitness(stats: dict) -> float:
    """Turn game stats into a single 'how good was this network' number.

    This is the MOST IMPORTANT design choice in evolution. The network will
    become good at *exactly* whatever you reward here. Reward the wrong thing
    and you get a network that does the wrong thing perfectly.

    Here: mostly reward hitting the ball (rallies), big bonus for scoring.
    """
    return stats["hits"] * 1.0 + stats["scored"] * 10.0


def evaluate(network: NeuralNetwork, n_games: int = 3) -> float:
    """Play several games and average the fitness.

    Why several? Because the ball starts randomly. A network might get lucky
    in one game. Averaging over a few games gives a fairer measure of skill.
    """
    game = PongGame()
    scores = [fitness(game.play_episode(network)) for _ in range(n_games)]
    return float(np.mean(scores))


def train(generations: int, population_size: int, elite_frac: float = 0.2,
          mutation_rate: float = 0.15, mutation_strength: float = 0.5) -> NeuralNetwork:
    """The main evolution loop."""

    # ── Step 1: Create a population of random networks ──
    population = [NeuralNetwork(LAYER_SIZES) for _ in range(population_size)]
    n_elite = max(2, int(population_size * elite_frac))

    best_ever = None
    best_ever_fitness = -np.inf

    logger.info(f"Starting evolution: {population_size} networks, {generations} generations")
    logger.info(f"Network shape: {LAYER_SIZES}\n")

    for gen in range(generations):
        # ── Step 2: Evaluate everyone ──
        scored = [(net, evaluate(net)) for net in population]

        # ── Step 3: Sort by fitness, best first (SELECTION) ──
        scored.sort(key=lambda x: x[1], reverse=True)

        best_fitness = scored[0][1]
        avg_fitness = np.mean([s for _, s in scored])

        # Track the best network we've ever seen
        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever = scored[0][0].copy()

        logger.info(f"Gen {gen+1:3d} | best: {best_fitness:6.1f} | avg: {avg_fitness:6.1f}")

        # ── Step 4: Make the next generation ──
        # Keep the elite (best performers) unchanged — 'elitism' ensures we
        # never lose our best solution to a bad mutation.
        elites = [net for net, _ in scored[:n_elite]]
        next_population = [net.copy() for net in elites]

        # Fill the rest with mutated children of the elites
        while len(next_population) < population_size:
            # Pick two parents from the elite (tournament-ish selection)
            parent_a = elites[np.random.randint(n_elite)]
            parent_b = elites[np.random.randint(n_elite)]

            # Crossover: mix the two parents
            child = NeuralNetwork.crossover(parent_a, parent_b)

            # Mutation: tweak the child randomly
            child.mutate(rate=mutation_rate, strength=mutation_strength)

            next_population.append(child)

        population = next_population

    logger.info(f"\nDone! Best fitness ever: {best_ever_fitness:.1f}")

    # Save the champion
    best_ever.save("champion_evolution.npz")
    logger.info("Saved best network to champion_evolution.npz")

    return best_ever


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pong AI with neuroevolution")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    args = parser.parse_args()

    champion = train(
        generations=args.generations,
        population_size=args.population,
        mutation_rate=args.mutation_rate,
    )

    # Show how the champion does
    logger.info("\nTesting champion over 10 games:")
    game = PongGame()
    for i in range(10):
        stats = game.play_episode(champion)
        logger.info(f"  Game {i+1}: {stats['hits']} hits, {stats['scored']} points scored")


if __name__ == "__main__":
    main()
