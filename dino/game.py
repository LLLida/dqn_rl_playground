import numpy as np
import random
import pygame

from typing import Tuple, Optional

class DinoGame:

    def __init__(self, max_steps: int = 300):
        self.width = 640
        self.height = 480

        self.state_dim = 6

        self.x = 50

        self.game_speed = 12
        self.player_width = 50
        self.player_height = 80
        self.obstacle_width = 50
        self.obstacle_height = 80
        self.obstacle_spawn_prob = 0.025
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

        self._spawn_obstacle_if_can()

        for obstacle in self.obstacles:
            dx = self.game_speed
            obstacle.x -= dx
        # remove obstacles out of bounds
        self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.x+obstacle.w >= 0]
        terminated = False
        for o in self.obstacles:
            if ((self.x >= o.x and self.x <= o.x + o.w) or (o.x >= self.x and o.x <= self.x + self.player_width)) and \
               ((self.y >= o.y and self.y <= o.y + o.h) or (o.y >= self.y and o.y <= self.y + self.player_height)):
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
        for o in self.obstacles:
            pygame.draw.rect(win, oc, (o.x, self.height-ground_height-o.h-o.y, o.w, o.h))
        pygame.draw.rect(win, pc, (self.x, self.height-self.player_height-ground_height-self.y, self.player_width, self.player_height))

    def _spawn_obstacle_if_can(self):
        left_margin = 250
        if (len(self.obstacles) == 0 or self.obstacles[-1].x < self.width-left_margin) and random.uniform(0, 1) <= self.obstacle_spawn_prob:
            x = self.width-1
            y = 0
            w = self.obstacle_width + random.randrange(-10, 10)
            h = self.obstacle_height + random.randrange(-5, 5)
            self.obstacles.append(Obstacle(x, y, w, h))

    def _state(self):
        nearest = Obstacle(-100, 0, 0, 0)
        for obstacle in self.obstacles:
            if obstacle.x > self.x:
                nearest = obstacle
                break
        return np.array([
            self.y / self.width, # координата динозаврика
            self.dy / 10.0,      # скорость динозаврика
            (nearest.x - self.x) / self.width, # координата x ближайшего препятствия
            (nearest.y - self.y) / self.height, # координата y ближайшего препятствия
            self.obstacle_width / self.width, # ширина ближайшего препятствия
            self.obstacle_height / self.height # высота ближайшего препятствия
        ])

class Obstacle:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
