"""
watch.py — Watch a trained network play Pong, rendered with pygame.

Run this after training to SEE your evolved network in action.

Usage:
    python watch.py                          # watch the evolution champion
    python watch.py --model champion_rl.npz  # watch a different saved network
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from network import NeuralNetwork
from pong import PongGame, WIDTH, HEIGHT, PADDLE_HEIGHT

# Scale game units to screen pixels
SCALE = 6
SCREEN_W = int(WIDTH * SCALE)
SCREEN_H = int(HEIGHT * SCALE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="champion_evolution.npz")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    try:
        import pygame
    except ImportError:
        print("pygame not installed. Run: pip install pygame")
        sys.exit(1)

    # Load the trained network. It could be an evolution net (NeuralNetwork,
    # [5,8,1]) or a DQN Q-network (QNetwork, [5,8,3]). We peek at the saved
    # file's 'kind' marker to pick the right loader; both expose .decide().
    try:
        saved = np.load(args.model)
        if "kind" in saved and str(saved["kind"]) == "dqn":
            from train_rl import QNetwork
            network = QNetwork.load(args.model)
            print(f"Loaded DQN model {args.model}")
        else:
            network = NeuralNetwork.load(args.model)
            print(f"Loaded {args.model}")
    except FileNotFoundError:
        print(f"No model found at {args.model}. Train one first:")
        print("  python train_evolution.py   (or)   python train_rl.py")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("NeuroPong — watch the AI play")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    game = PongGame()
    game.reset()

    running = True
    games_played = 0
    total_hits = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ── The network decides, the game advances ──
        state = game.get_state()
        action = network.decide(state)
        _, _, done = game.step(action)

        if done:
            games_played += 1
            total_hits += game.hits
            game.reset()

        # ── Draw everything ──
        screen.fill((15, 15, 30))

        # Center line
        for y in range(0, SCREEN_H, 20):
            pygame.draw.rect(screen, (40, 40, 60), (SCREEN_W // 2 - 1, y, 2, 10))

        # AI paddle (left, purple)
        pygame.draw.rect(screen, (127, 119, 221), (
            5,
            int((game.left_paddle_y - PADDLE_HEIGHT / 2) * SCALE),
            8,
            int(PADDLE_HEIGHT * SCALE),
        ))

        # Opponent paddle (right, gray)
        pygame.draw.rect(screen, (120, 120, 130), (
            SCREEN_W - 13,
            int((game.right_paddle_y - PADDLE_HEIGHT / 2) * SCALE),
            8,
            int(PADDLE_HEIGHT * SCALE),
        ))

        # Ball (teal)
        pygame.draw.circle(screen, (93, 202, 165), (
            int(game.ball_x * SCALE),
            int(game.ball_y * SCALE),
        ), 5)

        # Stats
        avg_hits = total_hits / games_played if games_played else 0
        txt = font.render(f"AI (left) | rally: {game.hits}  avg: {avg_hits:.1f}  games: {games_played}", True, (200, 200, 210))
        screen.blit(txt, (10, 10))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
