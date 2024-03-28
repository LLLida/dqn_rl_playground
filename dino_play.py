import pygame

from dino.game import DinoGame

game = DinoGame()

pygame.init()
win = pygame.display.set_mode((game.width, game.height))

pygame.display.set_caption('Dino')

game.reset()
clock = pygame.time.Clock()
running = True
paused = False
points = 0
while running:
    clock.tick(10)

    action = 0

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                action = 1
            if event.key == pygame.K_p:
                paused = not paused
            if event.key == pygame.K_i:
                print(state)
        if event.type == pygame.QUIT:
            running = False

    if paused:
        continue

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

    pygame.display.flip()
pygame.quit()
