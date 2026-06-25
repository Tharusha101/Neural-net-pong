"""
network.py — A neural network built FROM SCRATCH in NumPy.

No PyTorch, no TensorFlow. This is the whole thing, and it's small enough
to understand completely. Read every line — that's the point.

A neural network is just: take some numbers in, multiply by weights, add
biases, squish through a curve, repeat. That's it. The "intelligence" is
entirely in *what the weights are*. Evolution (later) is just a way of
searching for good weights.
"""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: turns negatives into 0, leaves positives alone.

    This is the 'squish' that makes a network nonlinear. Without an
    activation function, stacking layers would just be one big matrix
    multiply (i.e. a straight line) and the network couldn't learn curves.

    relu(-3) = 0,  relu(0) = 0,  relu(5) = 5
    """
    return np.maximum(0, x)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation: squishes any number into the range (-1, 1).

    Useful for outputs because it's bounded. We'll use it on the output
    layer so the paddle action is always between -1 (down) and +1 (up).
    """
    return np.tanh(x)


class NeuralNetwork:
    """A simple feedforward neural network.

    'Feedforward' means data flows one direction: input -> hidden -> output.
    No loops, no memory. Each layer is a matrix of weights + a vector of biases.

    Think of it like this for Pong:
        INPUTS (what the bird/paddle 'sees'):
            - ball x position
            - ball y position
            - ball x velocity
            - ball y velocity
            - paddle y position
        ↓ (multiply by weights of layer 1, add bias, apply ReLU)
        HIDDEN LAYER (the network's 'thinking' space)
        ↓ (multiply by weights of layer 2, add bias, apply tanh)
        OUTPUT:
            - one number: move paddle up (+) or down (-)
    """

    def __init__(self, layer_sizes: list[int], weights: list[np.ndarray] | None = None,
                 biases: list[np.ndarray] | None = None) -> None:
        """
        layer_sizes: e.g. [5, 8, 1] means 5 inputs, 8 hidden neurons, 1 output.

        If weights/biases aren't provided, we start with RANDOM ones.
        A randomly-initialized network plays terribly — that's expected.
        Evolution/training is the process of improving these random numbers.
        """
        self.layer_sizes = layer_sizes

        if weights is not None and biases is not None:
            # Use provided weights (e.g. from a parent during evolution)
            self.weights = weights
            self.biases = biases
        else:
            # Initialize random weights and biases
            self.weights = []
            self.biases = []
            for i in range(len(layer_sizes) - 1):
                in_size = layer_sizes[i]
                out_size = layer_sizes[i + 1]

                # Weights: a matrix connecting every neuron in this layer
                # to every neuron in the next. Shape: (out_size, in_size).
                # We scale by sqrt(2/in_size) — "He initialization" — which
                # keeps the signal from exploding or vanishing early on.
                w = np.random.randn(out_size, in_size) * np.sqrt(2.0 / in_size)

                # Biases: one per output neuron, start at zero.
                # Bias lets a neuron fire even when all inputs are zero —
                # it shifts the activation threshold.
                b = np.zeros(out_size)

                self.weights.append(w)
                self.biases.append(b)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run inputs through the network and get an output.

        This is the ENTIRE 'thinking' process of the network. It's called
        'forward' because data moves forward through the layers.

        For each layer:
            output = activation( weights @ input + bias )

        where @ is matrix multiplication.
        """
        x = np.asarray(inputs, dtype=np.float64)

        # Pass through every layer except the last using ReLU
        for i in range(len(self.weights) - 1):
            # weights[i] @ x  → matrix multiply: each hidden neuron computes
            # a weighted sum of all inputs. Then add bias, then squish.
            x = relu(self.weights[i] @ x + self.biases[i])

        # Last layer uses tanh so output is bounded in (-1, 1)
        x = tanh(self.weights[-1] @ x + self.biases[-1])

        return x

    def decide(self, inputs: np.ndarray) -> int:
        """Convert the network's raw output into a concrete action.

        Output > 0.1  → move up   (return +1)
        Output < -0.1 → move down (return -1)
        otherwise     → stay      (return 0)
        """
        output = self.forward(inputs)[0]
        if output > 0.1:
            return 1
        elif output < -0.1:
            return -1
        return 0

    # ─── Evolution helpers (used in train_evolution.py) ──────────────────

    def copy(self) -> "NeuralNetwork":
        """Make an exact clone of this network."""
        return NeuralNetwork(
            self.layer_sizes,
            weights=[w.copy() for w in self.weights],
            biases=[b.copy() for b in self.biases],
        )

    def mutate(self, rate: float = 0.1, strength: float = 0.5) -> None:
        """Randomly tweak some weights — this is how evolution explores.

        rate: fraction of weights to change (0.1 = 10% of them)
        strength: how big the random change is

        Mutation is the 'variation' in evolution. A mutated child might play
        slightly better or slightly worse than its parent. The ones that play
        better survive and reproduce. Over generations, good weights accumulate.
        """
        for w in self.weights:
            mask = np.random.random(w.shape) < rate
            w += mask * np.random.randn(*w.shape) * strength
        for b in self.biases:
            mask = np.random.random(b.shape) < rate
            b += mask * np.random.randn(*b.shape) * strength

    @staticmethod
    def crossover(parent_a: "NeuralNetwork", parent_b: "NeuralNetwork") -> "NeuralNetwork":
        """Combine two parents into a child (sexual reproduction for networks).

        For each weight, randomly take it from parent A or parent B. The idea:
        if both parents are good, mixing their 'genes' might combine their
        strengths. (This is optional — pure mutation also works.)
        """
        child = parent_a.copy()
        for i in range(len(child.weights)):
            mask = np.random.random(child.weights[i].shape) < 0.5
            child.weights[i] = np.where(mask, parent_a.weights[i], parent_b.weights[i])
            mask_b = np.random.random(child.biases[i].shape) < 0.5
            child.biases[i] = np.where(mask_b, parent_a.biases[i], parent_b.biases[i])
        return child

    # ─── Save / load ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the network's weights to a file."""
        data = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            data[f"w{i}"] = w
            data[f"b{i}"] = b
        data["layer_sizes"] = np.array(self.layer_sizes)
        np.savez(path, **data)

    @staticmethod
    def load(path: str) -> "NeuralNetwork":
        """Load a network from a file."""
        data = np.load(path)
        layer_sizes = data["layer_sizes"].tolist()
        n_layers = len(layer_sizes) - 1
        weights = [data[f"w{i}"] for i in range(n_layers)]
        biases = [data[f"b{i}"] for i in range(n_layers)]
        return NeuralNetwork(layer_sizes, weights=weights, biases=biases)
