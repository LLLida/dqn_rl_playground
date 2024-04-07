import numpy as np
import random
import pygame

from typing import Tuple, Optional

class DinoGame:

    def __init__(self, max_steps: int = 300):
        self.width = 640
        self.height = 480

        self.state_dim = 6

        self.x = 0.07825

        self.game_speed = 0.01875
        self.player_width = 0.07825
        self.player_height = 0.1667
        self.ground_obstacle_width = 0.07825
        self.ground_obstacle_height = 0.1667
        self.flying_obstacle_width = 0.0725
        self.flying_obstacle_height = 0.1
        self.obstacle_spawn_prob = 0.025
        self.obstacle_is_flying_prob = 0.25
        self.jump_a = 0.0417
        self.G = 0.00417
        self.max_steps = max_steps

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool]:
        self.t += 1
        reward = 0.02

        if action == 1 and self.y == 0:
            self.dy += self.jump_a
            if self.dy > 0:
                reward -= 0.03

        self.y += self.dy
        if self.y <= 0.:
            self.dy = 0.
            self.y = 0.
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

        self.t = 0.
        self.y = 0.
        self.dy = 0.
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
            pygame.draw.rect(win, oc, (o.x*self.width, self.height-ground_height-(o.h+o.y)*self.height, o.w*self.width, o.h*self.height))
        pygame.draw.rect(win, pc, (
            self.x*self.width,  # x
            self.height-ground_height-(self.y + self.player_height)*self.height, # y
            self.player_width*self.width, # w
            self.player_height*self.height # h
        ))

    def _spawn_obstacle_if_can(self):
        if (len(self.obstacles) == 0 or self.obstacles[-1].x < 0.3) and random.uniform(0, 1) <= self.obstacle_spawn_prob:
            x = 0.999
            if random.uniform(0, 1) <= self.obstacle_is_flying_prob:
                # спавним летающее препятствие
                y = self.player_height + 0.025
                w = self.flying_obstacle_width + random.uniform(-0.0125, 0.0125)
                h = self.flying_obstacle_height + random.uniform(-0.008, 0.008)
            else:
                # спавним наземное препятствие
                y = 0
                w = self.ground_obstacle_width + random.uniform(-0.015, 0.015)
                h = self.ground_obstacle_height + random.uniform(-0.01, 0.01)
            self.obstacles.append(Obstacle(x, y, w, h))

    def _state(self):
        nearest = Obstacle(-1, 0, 0, 0)
        for obstacle in self.obstacles:
            if obstacle.x > self.x:
                nearest = obstacle
                break
        return np.array([
            self.y, # координата динозаврика
            self.dy,      # скорость динозаврика
            nearest.x - self.x, # координата x ближайшего препятствия
            nearest.y - self.y, # координата y ближайшего препятствия
            nearest.w, # ширина ближайшего препятствия
            nearest.h # высота ближайшего препятствия
        ])

class Obstacle:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
