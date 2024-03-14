import pygame
import matplotlib.pyplot as plt
import numpy as np

from checkers.game import CheckersGame

checkers = CheckersGame()

pygame.init()

cell_size = 40
width, height = 8*cell_size, 8*cell_size
win = pygame.display.set_mode((width-cell_size, height))
pygame.display.set_caption('checkers')

bg_black = (0,0,0)
bg_white = (255,255,255)
fg_black = (50,50,50)
fg_white = (205,205,205)

clock = pygame.time.Clock()

running = True
while running:
    clock.tick(10)

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        if event.type == pygame.QUIT:
            running = False

    win.fill(bg_black)
    for i in range(8):
        for j in range(8):
            x, y = i, 7-j
            if (i+j)%2 == 1:
                pygame.draw.rect(win, bg_white, (x*cell_size, y*cell_size, cell_size, cell_size))

            piece = checkers.board[j, i]
            radius = cell_size//3
            if piece == 1:
                pygame.draw.circle(win, fg_white, (x*cell_size+cell_size//2, y*cell_size+cell_size//2), radius)
            if piece == -1:
                pygame.draw.circle(win, fg_black, (x*cell_size+cell_size//2, y*cell_size+cell_size//2), radius)
    pygame.display.flip()

pygame.quit()
