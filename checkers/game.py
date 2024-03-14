import numpy as np

from typing import Tuple, Optional

class CheckersGame:

    def __init__(self, max_steps: int = 300):
        self.max_steps = max_steps
        self.reset()

    def reset(self, seed:Optional[int] = None) -> np.ndarray:
        self.board = np.zeros((8, 8))

        # fill the board
        for i in range(12):
            # white piece
            self.board[i//4, 2*(i%4) + (i//4)%2] = 1
            # black piece
            self.board[7-i//4, 7-(2*(i%4) + (i//4)%2)] = -1
        print(self.board)

        return self.board
