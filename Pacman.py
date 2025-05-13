# libraries
import pygame
import time
import random
import numpy as np

# grid and display settings
GRID_SIZE = 10
CELL_SIZE = 60
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 60
TEST_NUMBER = 50
NUM_COINS = 5
NUM_WALLS = 8

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 100, 255)
RED = (255, 0, 0)

# Directions
DIRS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}

# list of moves
directions_list = list(np.load("Saved Data/predictions.npy"))
# print(directions_list)
directions_list = directions_list[:TEST_NUMBER]

# game settings
walls = set()
coins = set()
score = 0  # Global score variable


# player position
player_pos = [5, 5]

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pacman With EEG Signal")
clock = pygame.time.Clock()

# functions
## draw grid
def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, WHITE, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, WHITE, (0, y), (WIDTH, y))

## draw maze (with blue color)
def draw_maze():
    for (x, y) in walls:
        pygame.draw.rect(screen, BLUE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

## draw player (with yellow circle for pacman)
def draw_player():
    center_x = player_pos[0] * CELL_SIZE + CELL_SIZE // 2
    center_y = player_pos[1] * CELL_SIZE + CELL_SIZE // 2
    radius = CELL_SIZE // 2 - 5  # Slightly smaller than cell size
    pygame.draw.circle(screen, YELLOW, (center_x, center_y), radius)

## generate walls
def generate_walls():
    global walls
    walls.clear()
    # Create all possible positions except player starting position
    player_start = (player_pos[0], player_pos[1])
    available_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) 
                         if (x, y) != player_start]
    walls.update(random.sample(available_positions, NUM_WALLS))
    
## generate coins
def generate_coins():
    global coins
    coins.clear()
    player_start = (player_pos[0], player_pos[1])
    available_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) 
                         if (x, y) not in walls and (x, y) != player_start]  # exclude walls and player position
    coins.update(random.sample(available_positions, NUM_COINS))
    
## draw coins (with white circles)
def draw_coins():
    for x, y in coins:
        center_x = x * CELL_SIZE + CELL_SIZE // 2
        center_y = y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 4  # Make it smaller than pacman
        pygame.draw.circle(screen, WHITE, (center_x, center_y), radius)

## draw score
def draw_score():
    font = pygame.font.SysFont(None, 36)
    score_text = f"Score: {score}/{NUM_COINS}"
    text = font.render(score_text, True, RED)
    # Center the text at the top of the screen
    text_width = text.get_width()
    screen.blit(text, (WIDTH//2 - text_width//2, 10))

## move player
def move_player(direction):
    dx, dy = DIRS.get(direction, (0, 0))
    new_x = player_pos[0] + dx
    new_y = player_pos[1] + dy
    print(f"\nCurrent Direction: {direction}")
    print(f"Current Position: ({player_pos[0]}, {player_pos[1]})")
    print(f"Attempting to move to: ({new_x}, {new_y})")
    
    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
        if (new_x, new_y) not in walls:
            print("Move successful!")
            player_pos[0], player_pos[1] = new_x, new_y
        else:
            print("Cannot move - wall in the way")
    else:
        print("Cannot move - out of grid bounds")

# Initial generation
generate_walls()
generate_coins()

# game loop
running = True
move_index = 0
last_move_time = time.time()
score = 0

while running:
    screen.fill(BLACK)
    # draw_grid()
    draw_maze()
    draw_coins()
    draw_player()
    draw_score()
    
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False    # move every second based on the direction list
    if move_index < len(directions_list) and time.time() - last_move_time > 1:
        old_pos = tuple(player_pos)  # Remember position before moving
        move_player(directions_list[move_index])        # Check if player collected a coin at the new position
        new_pos = tuple(player_pos)
        if new_pos in coins and new_pos != old_pos:  # Only collect if we actually moved to the coin
            coins.remove(new_pos)
            score += 1
            print(f"Coin collected! Score: {score}/{NUM_COINS}")
        move_index += 1
        last_move_time = time.time()
    
    # Check for win condition (all coins collected)
    if not coins:  # if coins set is empty
        font = pygame.font.SysFont(None, 48)
        text = font.render("All Coins Collected!", True, YELLOW)
        screen.blit(text, (WIDTH//3, HEIGHT//2))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()    