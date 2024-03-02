import pygame
import random
import numpy as np

pygame.init()

game_width, game_height = 12, 26
cell_size = 30
top_margin = 4                  # чтобы фигуры плавно входили на экран
game_grid = np.zeros((top_margin+game_height, game_width), dtype=np.int32)

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

def spawn_piece():
    sx, sy = game_width//2, top_margin-1
    shape = random.choice(piece_shapes)

    blocks = []
    for i in range(len(shape)):
        for j in range(len(shape[i])):
            if shape[i][j] != 0:
                blocks.append((sx+j, sy+i))
    return blocks

def piece_pos(piece):
    x = sum([b[0] for b in piece]) / len(piece)
    y = sum([b[1] for b in piece]) / len(piece)
    return int(x), int(y)

def rotate_piece(piece):
    mx, my = piece_pos(piece)

    new_piece = [(mx + (y-my), my + (mx-x)) for x, y in piece]
    for x, y in new_piece:
        if x >= game_width or x < 0 or y >= game_height:
            return
        # if not (x, y) in new_piece and game_grid[y, x] != 0:
        #     return

    for i in range(len(piece)):
        x, y = piece[i]
        game_grid[y, x] = 0

    for i in range(len(piece)):
        x, y = piece[i]
        piece[i] = (mx + (y-my), my + (mx-x))

    for i in range(len(piece)):
        x, y = piece[i]
        game_grid[y, x] = 3

def move_piece(piece, dx, dy) -> bool:
    min_x = np.min([b[0] for b in piece])
    max_x = np.max([b[0] for b in piece])
    max_y = np.max([b[1] for b in piece])
    if max_x+dx > game_width-1 or min_x+dx < 0:
        return False
    if max_y >= game_height+top_margin-1:
        return True

    can_move = True
    for x, y in piece:
        if not (x+dx, y+dy) in piece and game_grid[y+dy, x+dx] != 0:
            can_move = False
            break
    if can_move:
        for i in range(len(piece)):
            x, y = piece[i]
            game_grid[y, x] = 0
            piece[i] = x+dx, y+dy

        for i in range(len(piece)):
            x, y = piece[i]
            game_grid[y, x] = 3

    return not can_move

piece = spawn_piece()

width, height = game_width*cell_size, game_height*cell_size
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

def draw_grid():
    for i in range(game_grid.shape[0]-top_margin):
        for j in range(game_grid.shape[1]):
            val = game_grid[i+top_margin, j]
            if val != 0:
                pygame.draw.rect(win, colors[val-1], (j*cell_size, i*cell_size, cell_size, cell_size))

clock = pygame.time.Clock()

running = True
paused = False
while running:
    clock.tick(10)

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_UP:
                rotate_piece(piece)
            if event.key == pygame.K_DOWN:
                move_piece(piece, 0, 1)
            if event.key == pygame.K_LEFT:
                move_piece(piece, -1, 0)
            if event.key == pygame.K_RIGHT:
                move_piece(piece, 1, 0)
            if event.key == pygame.K_SPACE:
                paused = not paused
        if event.type == pygame.QUIT:
            running = False

    if paused:
        continue

    if move_piece(piece, 0, 1):
        piece = spawn_piece()

        points = 0
        # проверяем заполнилась ли какая-то строчка
        i = game_grid.shape[0]-1
        while i >= top_margin:
            row = game_grid[i, :]
            if np.sum(row) == 0:
                break
            if np.all(row > 0):
                points += 10
                game_grid[top_margin:i+1, :] = game_grid[top_margin-1:i, :]
            else:
                i -= 1
        if points>0: print(f'Got {points} points!')

        if np.sum(game_grid[top_margin]) > 0:
            print('Game over!')
            paused = True

    # print(piece)

    win.fill(black)

    draw_grid()

    pygame.display.flip()

pygame.quit()
