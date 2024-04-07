import numpy as np
import pygame
import torch

from dino.game import DinoGame
from dino.agent import DinoAgent

game = DinoGame()
agent = DinoAgent(game.state_dim, epsilon=0.0)
agent_path = 'dino.pkl'
try:
    agent.load_state_dict(torch.load(agent_path))
except:
    print(f'Файл нейронной сети не нашёлся в "{agent_path}". Используем рандомные веса')

pygame.init()
win = pygame.display.set_mode((game.width, game.height))

font = pygame.font.SysFont('Arrial', 40)

pygame.display.set_caption('Dino')

prev_state = game.reset()

clock = pygame.time.Clock()
running = True
paused = False
points = 0
human_playing = True
while running:
    clock.tick(10)

    action = 0

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                action = 1
            if event.key == pygame.K_RETURN: # при нажатии enter меняем игрока(человек или нейронка)
                human_playing = not human_playing
            if event.key == pygame.K_p:
                paused = not paused
            if event.key == pygame.K_i:
                print(state)
        if event.type == pygame.QUIT:
            running = False

    if paused:
        continue

    if not human_playing:
        qvalues = agent.get_qvalues(prev_state[np.newaxis])
        action  = agent.sample_actions(qvalues)[0]
        text = f'qvalues: [{qvalues[0, 0]:.3f}, {qvalues[0, 1]:.3f}]'
    else:
        text = 'press SPACE to jump, ENTER to switch player'

    text_surface = font.render(text, False, (255, 255, 255))

    state, reward, terminated, _ = game.step(action)
    points += reward
    if terminated:
        print(f'You scored {points} points!')
        paused = True
        points = 0
        game.render(win)
        state = game.reset()
    else:
        game.render(win)

    prev_state = state

    win.blit(text_surface, (0,0))

    pygame.display.flip()
pygame.quit()
