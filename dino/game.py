import numpy as np
import random
import pygame

from typing import Tuple, Optional

class DinoGame:

    def __init__(self, max_steps: int = 300):
        self.width = 640
        self.height = 480

        self.player_height = 80
        self.obs_prob = 0.05
        self.max_steps = max_steps

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool]:
        if action == 1 and self.y == 0:
            self.dy += 12

        self.y += self.dy
        if self.y <= 0:
            self.dy = 0
        else:
            self.dy -= 1

        if self._can_spawn_obstacle():
            x = self.width-1
            y = 0
            self.obstacles.append((x, y))
        for i, (x, y) in enumerate(self.obstacles):
            dx = 12
            self.obstacles[i] = (x-dx, y)
        # remove obstacles out of bounds
        self.obstacles = [(x, y) for x, y in obstacles if x+50 < 0]

        self.t += 1
        reward = 0.1 * (t % 5 == 0)
        # TODO: iterate all obstacles for intersection
        truncated = t >= self.max_steps

        return None, reward, None, truncated

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        random.seed(seed)

        self.t = 0
        self.y = 0
        self.dy = 0
        self.obstacles = []

    def render(self, win):
        bg = (0, 0, 0)
        pc = (255, 255, 255)
        oc = (255, 0, 0)
        ground = (100, 100, 20)
        player_height = 100
        scale = 2

        win.fill(bg)
        pygame.draw.rect(win, ground, (0, self.height-50, self.width, 50))
        for x, y in self.obstacles:
            pygame.draw.rect(win, oc, (x, self.height-50-player_height-y, 50, player_height))
        pygame.draw.rect(win, pc, (50, self.height-player_height-(50+scale*self.y), 50, player_height))

    def _can_spawn_obstacle(self):
        left_margin = 250
        return (len(self.obstacles) == 0 or self.obstacles[-1][0] < self.width-left_margin) and random.uniform(0, 1) <= self.obs_prob
