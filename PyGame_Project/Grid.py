import pygame
import numpy as np
import random
from pygame.locals import *

# Constants
GRID_SIZE = 6
CELL_SIZE = 80
MARGIN = 50
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE * 2 + MARGIN * 3
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + MARGIN * 2

# Colors
BACKGROUND = (30, 30, 50)
WALL_COLOR = (50, 50, 70)
PLAYER_COLOR = (70, 130, 180)
GOAL_COLOR = (50, 180, 80)
HAZARD_COLOR = (220, 80, 60)
TEXT_COLOR = (220, 220, 220)
Q_VALUE_COLOR = (180, 180, 250)

# Rewards
GOAL_REWARD = 100
HAZARD_PENALTY = -50
STEP_PENALTY = -1

# Maze layout (0=path, 1=wall, 2=hazard, 3=goal)
MAZE_LAYOUT = [
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 2, 0],
    [0, 0, 0, 0, 1, 3],
    [0, 1, 1, 0, 1, 0],
    [0, 2, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0]
]

class MazeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Maze Q-Learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
        
        self.grid = np.array(MAZE_LAYOUT)
        self.player_pos = [0, 0]
        self.goal_pos = self.find_goal()
        self.mode = "manual"  # or "training"
        self.running = True
        self.episode_count = 0
        
        # Q-learning parameters
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # (x, y, action)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.min_epsilon = 0.1
        self.training_speed = 10  # episodes per frame
    
    def find_goal(self):
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.grid[x, y] == 3:
                    return [x, y]
        return [0, 0]  # Fallback
    
    def reset(self):
        self.player_pos = [0, 0]
        return tuple(self.player_pos)
    
    def step(self, action):
        x, y = self.player_pos
        new_x, new_y = x, y
        
        # 0=up, 1=right, 2=down, 3=left
        if action == 0 and x > 0: new_x = x - 1
        elif action == 1 and y < GRID_SIZE-1: new_y = y + 1
        elif action == 2 and x < GRID_SIZE-1: new_x = x + 1
        elif action == 3 and y > 0: new_y = y - 1
        
        # Check if move is valid (not a wall)
        if self.grid[new_x, new_y] != 1:
            self.player_pos = [new_x, new_y]
        
        # Check if player reached goal or hazard
        cell_value = self.grid[self.player_pos[0], self.player_pos[1]]
        if cell_value == 3:  # Goal
            reward = GOAL_REWARD
            done = True
        elif cell_value == 2:  # Hazard
            reward = HAZARD_PENALTY
            done = True
        else:  # Path
            reward = STEP_PENALTY
            done = False
        
        if done:
            self.reset()
            self.episode_count += 1
        
        return tuple(self.player_pos), reward, done
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Random action
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])  # Best action
    
    def update_q_table(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        best_next = np.max(self.q_table[next_x, next_y])
        
        # Q-learning formula
        self.q_table[x, y, action] += self.learning_rate * (
            reward + self.discount_factor * best_next - self.q_table[x, y, action]
        )
    
    def train_episode(self):
        state = self.reset()
        done = False
        
        while not done:
            action = self.choose_action(state)
            next_state, reward, done = self.step(action)
            self.update_q_table(state, action, reward, next_state)
            state = next_state
        
        # Reduce exploration over time
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.999)
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_t:
                    self.mode = "training"
                elif event.key == K_m:
                    self.mode = "manual"
                elif event.key == K_r:
                    self.reset()
                    self.episode_count = 0
                    self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
                    self.epsilon = 0.9
                elif event.key == K_PLUS or event.key == K_EQUALS:
                    self.training_speed = min(100, self.training_speed + 5)
                elif event.key == K_MINUS:
                    self.training_speed = max(1, self.training_speed - 5)
                
                if self.mode == "manual":
                    if event.key == K_UP: self.step(0)
                    elif event.key == K_RIGHT: self.step(1)
                    elif event.key == K_DOWN: self.step(2)
                    elif event.key == K_LEFT: self.step(3)
    
    def update(self):
        if self.mode == "training":
            for _ in range(self.training_speed):
                self.train_episode()
    
    def draw(self):
        self.screen.fill(BACKGROUND)
        
        # Draw main grid (environment)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(
                    MARGIN + y * CELL_SIZE,
                    MARGIN + x * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
                
                # Cell color
                if self.grid[x, y] == 1:  # Wall
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)
                elif self.grid[x, y] == 2:  # Hazard
                    pygame.draw.rect(self.screen, HAZARD_COLOR, rect)
                elif self.grid[x, y] == 3:  # Goal
                    pygame.draw.rect(self.screen, GOAL_COLOR, rect)
                
                # Grid lines
                pygame.draw.rect(self.screen, (60, 60, 80), rect, 1)
                
                # Player
                if x == self.player_pos[0] and y == self.player_pos[1]:
                    pygame.draw.circle(
                        self.screen, PLAYER_COLOR,
                        (MARGIN + y * CELL_SIZE + CELL_SIZE//2, 
                         MARGIN + x * CELL_SIZE + CELL_SIZE//2),
                        CELL_SIZE//3
                    )
        
        # Draw Q-value grid
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(
                    MARGIN * 2 + GRID_SIZE * CELL_SIZE + y * CELL_SIZE,
                    MARGIN + x * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
                
                # Background
                pygame.draw.rect(self.screen, (40, 40, 60), rect)
                pygame.draw.rect(self.screen, (60, 60, 80), rect, 1)
                
                # Q-values
                max_q = np.max(self.q_table[x, y])
                if max_q != 0:  # Only draw if we have values
                    q_text = self.font.render(f"{max_q:.1f}", True, Q_VALUE_COLOR)
                    self.screen.blit(q_text, (rect.x + 5, rect.y + 5))
        
        # Draw stats
        mode_text = self.font.render(f"Mode: {self.mode}", True, TEXT_COLOR)
        episode_text = self.font.render(f"Episodes: {self.episode_count}", True, TEXT_COLOR)
        epsilon_text = self.font.render(f"Epsilon: {self.epsilon:.2f}", True, TEXT_COLOR)
        speed_text = self.font.render(f"Speed: {self.training_speed}", True, TEXT_COLOR)
        controls_text = self.font.render("Controls: T=Train, M=Manual, R=Reset", True, TEXT_COLOR)
        controls_text2 = self.font.render("+/-=Speed, Arrows=Move", True, TEXT_COLOR)
        
        self.screen.blit(mode_text, (MARGIN, WINDOW_HEIGHT - MARGIN + 10))
        self.screen.blit(episode_text, (MARGIN, WINDOW_HEIGHT - MARGIN + 30))
        self.screen.blit(epsilon_text, (MARGIN, WINDOW_HEIGHT - MARGIN + 50))
        self.screen.blit(speed_text, (MARGIN + 200, WINDOW_HEIGHT - MARGIN + 10))
        self.screen.blit(controls_text, (MARGIN + 200, WINDOW_HEIGHT - MARGIN + 30))
        self.screen.blit(controls_text2, (MARGIN + 200, WINDOW_HEIGHT - MARGIN + 50))
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    game = MazeGame()
    game.run()