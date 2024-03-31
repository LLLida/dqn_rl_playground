import numpy as np
import random
import pygame

from typing import Tuple, Optional

class DinoGame:

    def __init__(self, max_steps: int = 300):
        self.width = 640
        self.height = 480

        self.state_dim = 4

        self.x = 50

        self.game_speed = 12
        self.player_width = 50
        self.player_height = 80
        self.obstacle_width = 50
        self.obstacle_height = 80
        self.obstacle_spawn_prob = 0.02
        self.jump_a = 20
        self.G = 2
        self.max_steps = max_steps

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool]:
        self.t += 1
        reward = 0.02

        if action == 1 and self.y == 0:
            self.dy += self.jump_a
            if self.dy > 0:
                reward -= 0.03

        self.y += self.dy
        if self.y <= 0:
            self.dy = 0
        else:
            self.dy -= self.G

        if self._can_spawn_obstacle():
            x = self.width-1
            y = 0
            self.obstacles.append((x, y))
        for i, (x, y) in enumerate(self.obstacles):
            dx = self.game_speed
            self.obstacles[i] = (x-dx, y)
        # remove obstacles out of bounds
        self.obstacles = [(x, y) for x, y in self.obstacles if x+self.obstacle_width >= 0]
        terminated = False
        for (x, y) in self.obstacles:
            if ((self.x >= x and self.x <= x + self.obstacle_width) or (x >= self.x and x <= self.x + self.player_width)) and \
               ((self.y >= y and self.y <= y + self.obstacle_height) or (y >= self.y and y <= self.y + self.player_height)):
                # штрафуем за то, что агент врезался в препятствие
                reward -= 5
                terminated = True
                break
        truncated = self.t >= self.max_steps

        return self._state(), reward, terminated, truncated

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        random.seed(seed)

        self.t = 0
        self.y = 0
        self.dy = 0
        self.obstacles = []

        return self._state()

    def render(self, win):
        bg = (0, 0, 0)
        pc = (255, 255, 255)
        oc = (255, 0, 0)
        ground = (100, 100, 20)
        ground_height = 50

        win.fill(bg)
        pygame.draw.rect(win, ground, (0, self.height-ground_height, self.width, ground_height))
        for x, y in self.obstacles:
            pygame.draw.rect(win, oc, (x, self.height-ground_height-self.obstacle_height-y, self.obstacle_width, self.obstacle_height))
        pygame.draw.rect(win, pc, (self.x, self.height-self.player_height-ground_height-self.y, self.player_width, self.player_height))

    def _can_spawn_obstacle(self):
        left_margin = 250
        return (len(self.obstacles) == 0 or self.obstacles[-1][0] < self.width-left_margin) and random.uniform(0, 1) <= self.obstacle_spawn_prob

    def _state(self):
        rightmost = (-100, 0)
        for (x, y) in self.obstacles:
            if x > self.x:
                rightmost = (x, y)
                break
        # state = np.array([self.y, self.dy, rightmost[0], rightmost[1]])
        return np.array([self.y / self.width, self.dy / 10.0, rightmost[0] / self.width, rightmost[1] / self.height])
