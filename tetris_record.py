import pickle
import pygame
import numpy as np

from tetris.game import TetrisGame
from rl.replay_buffer import ReplayBuffer

save_path = 'exp_replay.pkl'

tetris = TetrisGame(width=12, height=26)

exp_replay = ReplayBuffer(size=10000)

pygame.init()

cell_size = 30
width, height = tetris.width*cell_size, tetris.height*cell_size
win = pygame.display.set_mode((width, height))
pygame.display.set_caption('Tetris')

colors = [
    (0, 0, 0),
    (255, 255, 255),
    (255, 10, 10),
    (10, 255, 10),
    (10, 10, 255),
    (255, 255, 10),
    (10, 255, 255),
    (255, 10, 255)
]
black = colors[0]
white = colors[1]

clock = pygame.time.Clock()

prev_state = tetris.reset()

points = 0
running = True
paused = False
while running:
    clock.tick(10)

    action = 'nothing'

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_UP:
                action = 'up'
            if event.key == pygame.K_LEFT:
                action = 'left'
            if event.key == pygame.K_RIGHT:
                action = 'right'
            if event.key == pygame.K_SPACE: # space ставит игру на паузу
                paused = not paused
        if event.type == pygame.QUIT:
            running = False

    if paused:
        continue

    action = TetrisGame.action_names[action]

    state, reward, terminated, _ = tetris.step(action) # (ignore truncated flag)
    points += reward
    if terminated:
        print(f'You scored {points} points!')
        state = tetris.reset()
        paused = True

    exp_replay.add(prev_state, action, reward, state, done=terminated)

    prev_state = state
    grid = state[1]

    win.fill(black)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != 0:
                pygame.draw.rect(win, colors[grid[i, j]-1], (j*cell_size, i*cell_size, cell_size, cell_size))

    pygame.display.flip()
pygame.quit()

with open(save_path, 'wb') as f:
    pickle.dump(exp_replay, f)
    print(f'Игра сохранена в {save_path}... ({len(exp_replay)} шагов)')
