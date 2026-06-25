"""
pong.py — A minimal Pong environment.

The AI controls the left paddle. A simple scripted opponent controls the
right paddle (it just follows the ball, imperfectly). The game runs
'headless' (no graphics) during training for speed, and can be rendered
with pygame when you want to watch.

The key idea for ML: the game exposes a STATE (what the network sees) and
accepts an ACTION (what the network decides). That's the interface between
'game' and 'brain'.
"""

from __future__ import annotations

import numpy as np

# Game dimensions (arbitrary units)
WIDTH = 100.0
HEIGHT = 100.0
PADDLE_HEIGHT = 20.0
PADDLE_SPEED = 3.0
BALL_SPEED = 2.0


class PongGame:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        """Start a new game. Returns the initial state."""
        # Paddles start centered (y-position of paddle center)
        self.left_paddle_y = HEIGHT / 2
        self.right_paddle_y = HEIGHT / 2

        # Ball starts in the middle, moving in a random direction
        self.ball_x = WIDTH / 2
        self.ball_y = HEIGHT / 2
        angle = self.rng.uniform(-0.5, 0.5)  # mostly horizontal
        direction = 1 if self.rng.random() > 0.5 else -1
        self.ball_vx = direction * BALL_SPEED * np.cos(angle)
        self.ball_vy = BALL_SPEED * np.sin(angle)

        self.score_left = 0
        self.score_right = 0
        self.steps = 0
        self.hits = 0  # how many times the left (AI) paddle hit the ball

        return self.get_state()

    def get_state(self) -> np.ndarray:
        """What the AI 'sees'. All values normalized to roughly [-1, 1].

        Normalizing inputs matters! Neural nets work best when inputs are
        on a similar, small scale. Feeding raw pixel coords (0-100) makes
        training harder than feeding normalized values (-1 to 1).
        """
        return np.array([
            (self.ball_x / WIDTH) * 2 - 1,        # ball x: -1 (left) to +1 (right)
            (self.ball_y / HEIGHT) * 2 - 1,       # ball y: -1 (top) to +1 (bottom)
            self.ball_vx / BALL_SPEED,            # ball x velocity, normalized
            self.ball_vy / BALL_SPEED,            # ball y velocity, normalized
            (self.left_paddle_y / HEIGHT) * 2 - 1, # AI paddle y position
        ])

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Advance the game one tick.

        action: -1 (move down), 0 (stay), +1 (move up) — from the AI

        Returns (new_state, reward, done):
            reward: small signal of how good this step was (used by RL later)
            done: True if the game is over
        """
        self.steps += 1
        reward = 0.0

        # ── Move AI paddle ──
        self.left_paddle_y += action * PADDLE_SPEED
        self.left_paddle_y = np.clip(self.left_paddle_y, PADDLE_HEIGHT / 2, HEIGHT - PADDLE_HEIGHT / 2)

        # ── Move opponent paddle (scripted: follows ball imperfectly) ──
        # It only moves 70% as fast as it could, so it's beatable.
        target = self.ball_y
        diff = target - self.right_paddle_y
        self.right_paddle_y += np.clip(diff, -PADDLE_SPEED * 0.7, PADDLE_SPEED * 0.7)
        self.right_paddle_y = np.clip(self.right_paddle_y, PADDLE_HEIGHT / 2, HEIGHT - PADDLE_HEIGHT / 2)

        # ── Move ball ──
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Bounce off top and bottom walls
        if self.ball_y <= 0 or self.ball_y >= HEIGHT:
            self.ball_vy *= -1
            self.ball_y = np.clip(self.ball_y, 0, HEIGHT)

        done = False

        # ── Left wall: AI paddle zone ──
        if self.ball_x <= 2.0:
            # Did the AI paddle catch it?
            if abs(self.ball_y - self.left_paddle_y) <= PADDLE_HEIGHT / 2:
                self.ball_vx *= -1   # bounce back
                self.ball_x = 2.0
                self.hits += 1
                reward = 1.0          # GOOD: caught the ball
                # Add a bit of angle based on where it hit the paddle
                offset = (self.ball_y - self.left_paddle_y) / (PADDLE_HEIGHT / 2)
                self.ball_vy += offset * 0.5
            else:
                # Missed — AI loses the point
                self.score_right += 1
                reward = -1.0         # BAD: missed the ball
                done = True

        # ── Right wall: opponent paddle zone ──
        elif self.ball_x >= WIDTH - 2.0:
            if abs(self.ball_y - self.right_paddle_y) <= PADDLE_HEIGHT / 2:
                self.ball_vx *= -1
                self.ball_x = WIDTH - 2.0
                offset = (self.ball_y - self.right_paddle_y) / (PADDLE_HEIGHT / 2)
                self.ball_vy += offset * 0.5
            else:
                # AI scored against opponent!
                self.score_left += 1
                reward = 2.0          # GREAT: scored a point
                done = True

        # End game if it drags on too long (prevents infinite rallies)
        if self.steps >= 2000:
            done = True

        return self.get_state(), reward, done

    def play_episode(self, network, max_steps: int = 2000) -> dict:
        """Run one full game with a network controlling the AI paddle.

        Returns stats used to compute 'fitness' (how good this network is).
        This is what evolution uses to rank networks.
        """
        self.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            state = self.get_state()
            action = network.decide(state)
            _, reward, done = self.step(action)
            total_reward += reward
            if done:
                break

        return {
            "hits": self.hits,           # main fitness signal: rally length
            "scored": self.score_left,    # bonus: did it score?
            "steps_survived": self.steps,
            "total_reward": total_reward,
        }
