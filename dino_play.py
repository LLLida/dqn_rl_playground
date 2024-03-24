import pygame

from dino.game import DinoGame

game = DinoGame()

pygame.init()
win = pygame.display.set_mode((game.width, game.height))

pygame.display.set_caption('Dino')

game.reset()
clock = pygame.time.Clock()
running = True
while running:
    clock.tick(10)

    action = 0

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                action = 1
        if event.type == pygame.QUIT:
            running = False

    state, reward, terminated, _ = game.step(action)

    game.render(win)

    pygame.display.flip()
pygame.quit()
