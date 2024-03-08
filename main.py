import pygame
import random
import numpy as np

from tetris.game import TetrisGame
from tetris.agent import TetrisAgent
import training as T

pygame.init()

tetris = TetrisGame(width=12, height=26)
agent = TetrisAgent(ncols=12, nrows=26, nactions=4, epsilon=0.3)

# print(T.evaluate(tetris, agent)); exit(0)

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
human_playing = True
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
            if event.key == pygame.K_RETURN: # при нажатии enter меняем игрока(человек или нейронка)
                human_playing = not human_playing
        if event.type == pygame.QUIT:
            running = False

    if paused:
        continue

    if human_playing:
        action = TetrisGame.action_names[action]
    else:
        qvalues = agent.get_qvalues(prev_state[np.newaxis])
        action = agent.sample_actions(qvalues)[0]

    grid, reward, terminated, _ = tetris.step(action) # (ignore truncated flag)
    points += reward
    if terminated:
        print(f'You scored {points} points!')
        grid = tetris.reset()
        paused = True

    prev_state = grid
    grid = grid[1]

    win.fill(black)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != 0:
                pygame.draw.rect(win, colors[grid[i, j]-1], (j*cell_size, i*cell_size, cell_size, cell_size))

    pygame.display.flip()

pygame.quit()
