import numpy as np
import random

from typing import Tuple, Optional

class TetrisGame:

    action_names = {
        'nothing': 0,
        'left': 1,
        'right': 2,
        'up': 3
    }

    def __init__(self, width: int, height: int, max_steps: int = 300):
        self.width = width
        self.height = height
        self.top_margin = 4
        self.grid = np.zeros((self.top_margin+height, width), dtype=np.int32)
        self.step_id = 0
        self.max_steps = max_steps

        self._spawn_piece()

    def step(self, action: int) -> Tuple[np.ndarray, int, bool]:
        points = 0
        terminated = False

        prev_obs = self.grid[self.top_margin:]

        if action == 1:
            self._move_piece(-1, 0)
        if action == 2:
            self._move_piece(1, 0)
        if action == 3:
            self._rotate_piece()

        if self._move_piece(0, 1):
            self._spawn_piece()

            points = 0

            i = self.grid.shape[0]-1
            while i >= self.top_margin:
                row = self.grid[i, :]
                if np.sum(row) == 0:
                    break
                if np.all(row > 0):
                    points += 1
                    self.grid[self.top_margin:i+1, :] = self.grid[self.top_margin-1:i, :]
                else:
                    i -= 1

            if np.sum(self.grid[self.top_margin]) > 0:
                terminated = True

        obs = np.concatenate([prev_obs[np.newaxis, :, :], self.grid[np.newaxis, self.top_margin:, :]])

        truncated = False
        self.step_id += 1
        if self.step_id > self.max_steps:
            truncated = True

        return obs, points, terminated, truncated

    def reset(self, seed:Optional[int] = None) -> np.ndarray:
        self.step_id = 0
        random.seed(seed)

        self.grid = np.zeros((self.top_margin+self.height, self.width), dtype=np.int32)
        self._spawn_piece()

        obs = self.grid[np.newaxis, self.top_margin:, :]
        return np.concatenate([obs, obs])

    def _spawn_piece(self):
        piece_shapes = [
            # квадратик
            [[1, 1],
             [1, 1]],
            # ступенька вправо
            [[0, 1, 1],
             [1, 1]],
            # ступенька влево
            [[1, 1],
             [0, 1, 1]],
            # палочка
            [[1, 1, 1, 1]],
            # палочка с крюком справа
            [[0, 0, 1],
             [1, 1, 1]],
            # палочка с крюком слева
            [[1, 0, 0],
             [1, 1, 1]],
            # треугольник
            [[0, 1, 0],
             [1, 1, 1]]
        ]

        sx, sy = self.width//2, self.top_margin-1
        shape = random.choice(piece_shapes)

        self.piece = []
        for i in range(len(shape)):
            for j in range(len(shape[i])):
                if shape[i][j] != 0:
                    self.piece.append((sx+j, sy+i))

    def _piece_pos(self):
        x = sum([b[0] for b in self.piece]) / len(self.piece)
        y = sum([b[1] for b in self.piece]) / len(self.piece)
        return int(x), int(y)

    def _rotate_piece(self):
        mx, my = self._piece_pos()

        new_piece = [(mx + (y-my), my + (mx-x)) for x, y in self.piece]
        for x, y in new_piece:
            if x >= self.width or x < 0 or y >= self.height:
                return
            # if not (x, y) in new_piece and grid[y, x] != 0:
            #     return

        for i in range(len(self.piece)):
            x, y = self.piece[i]
            self.grid[y, x] = 0

        for i in range(len(self.piece)):
            x, y = self.piece[i]
            self.piece[i] = (mx + (y-my), my + (mx-x))

        for i in range(len(self.piece)):
            x, y = self.piece[i]
            self.grid[y, x] = 3

    def _move_piece(self, dx, dy) -> bool:
        min_x = np.min([b[0] for b in self.piece])
        max_x = np.max([b[0] for b in self.piece])
        max_y = np.max([b[1] for b in self.piece])
        if max_x+dx > self.width-1 or min_x+dx < 0:
            return False
        if max_y >= self.height+self.top_margin-1:
            return True

        can_move = True
        for x, y in self.piece:
            if not (x+dx, y+dy) in self.piece and self.grid[y+dy, x+dx] != 0:
                can_move = False
                break
        if can_move:
            for i in range(len(self.piece)):
                x, y = self.piece[i]
                self.grid[y, x] = 0
                self.piece[i] = x+dx, y+dy

            for i in range(len(self.piece)):
                x, y = self.piece[i]
                self.grid[y, x] = 3

        return not can_move
