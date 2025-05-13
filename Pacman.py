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
directions_list = list(np.load("predictions.npy"))
# print(directions_list)
directions_list = directions_list[:TEST_NUMBER]

# game settings
walls = set()
coins = set()
score = 0
game_started = False
game_mode = None  # 'prediction' or 'keyboard'

# Button settings
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 50
BUTTON_PADDING = 20

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

## draw buttons
def draw_buttons():
    font = pygame.font.SysFont(None, 36)
    
    # Prediction mode button
    pred_button_rect = pygame.Rect(WIDTH//2 - BUTTON_WIDTH - BUTTON_PADDING, HEIGHT//2 - BUTTON_HEIGHT//2, 
                                 BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(screen, BLUE, pred_button_rect)
    pred_text = font.render("Predictions", True, WHITE)
    pred_text_rect = pred_text.get_rect(center=pred_button_rect.center)
    screen.blit(pred_text, pred_text_rect)
    
    # Keyboard mode button
    key_button_rect = pygame.Rect(WIDTH//2 + BUTTON_PADDING, HEIGHT//2 - BUTTON_HEIGHT//2, 
                                BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(screen, BLUE, key_button_rect)
    key_text = font.render("Keyboard", True, WHITE)
    key_text_rect = key_text.get_rect(center=key_button_rect.center)
    screen.blit(key_text, key_text_rect)
    
    return pred_button_rect, key_button_rect

## handle button clicks
def handle_button_click(pos, pred_button_rect, key_button_rect):
    global game_started, game_mode
    if pred_button_rect.collidepoint(pos):
        game_started = True
        game_mode = 'prediction'
    elif key_button_rect.collidepoint(pos):
        game_started = True
        game_mode = 'keyboard'
        global NUM_COINS
        global NUM_WALLS
        NUM_COINS = 10
        NUM_WALLS = 15

    # Reset game state
    score = 0
    move_index = 0
    player_pos[0], player_pos[1] = 5, 5  # Reset player position
    generate_walls()
    generate_coins()

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
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
            # Reset game to main menu
            game_started = False
            game_mode = None
            score = 0
            move_index = 0
            last_move_time = time.time()
            player_pos[0], player_pos[1] = 5, 5
        elif e.type == pygame.MOUSEBUTTONDOWN and not game_started:
            # Handle button clicks in menu
            pred_button_rect, key_button_rect = draw_buttons()
            handle_button_click(e.pos, pred_button_rect, key_button_rect)
        elif e.type == pygame.KEYDOWN and game_mode == 'keyboard' and coins:  # Only move if there are coins left
            # Handle keyboard controls
            old_pos = tuple(player_pos)
            if e.key == pygame.K_UP:
                move_player("up")
            elif e.key == pygame.K_DOWN:
                move_player("down")
            elif e.key == pygame.K_LEFT:
                move_player("left")
            elif e.key == pygame.K_RIGHT:
                move_player("right")
            # Check for coin collection after keyboard move
            new_pos = tuple(player_pos)
            if new_pos in coins and new_pos != old_pos:
                coins.remove(new_pos)
                score += 1
                print(f"Coin collected! Score: {score}/{NUM_COINS}")
    
    if not game_started:
        # Draw menu
        pred_button_rect, key_button_rect = draw_buttons()
    else:
        # Draw game
        draw_maze()
        draw_coins()
        draw_player()
        draw_score()
        
        # Handle prediction-based movement
        if game_mode == 'prediction' and move_index < len(directions_list) and time.time() - last_move_time > 1:
            old_pos = tuple(player_pos)
            move_player(directions_list[move_index])
            new_pos = tuple(player_pos)
            if new_pos in coins and new_pos != old_pos:
                coins.remove(new_pos)
                score += 1
                print(f"Coin collected! Score: {score}/{NUM_COINS}")
            move_index += 1
            last_move_time = time.time()
        
        # Check for win condition
        if not coins:
            font = pygame.font.SysFont(None, 48)
            text = font.render("All Coins Collected!", True, YELLOW)
            screen.blit(text, (WIDTH//3, HEIGHT//2))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()    