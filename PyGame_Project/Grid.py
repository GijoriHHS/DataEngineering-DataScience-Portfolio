import pygame
import numpy as np
import random

# Constanten
GRID_SIZE = 6
CELL_SIZE = 80
MARGIN = 30
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE * 2 + MARGIN * 3
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + MARGIN * 2 + 80

# Kleuren
BACKGROUND = (30, 30, 50)
WALL_COLOR = (55, 55, 75)
PLAYER_COLOR = (70, 130, 180)
GOAL_COLOR = (50, 180, 80)
HAZARD_COLOR = (220, 80, 60)
TEXT_COLOR = (220, 220, 220)
ARROW_COLOR = (255, 255, 100)

# Maze (0=pad, 1=muur, 2=gevaar, 3=doel)
MAZE = [
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 2, 0],
    [0, 0, 0, 0, 1, 3],
    [0, 1, 1, 0, 1, 0],
    [0, 2, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0]
]

class SimpleMaze:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Q-Learning Maze")
        self.font = pygame.font.SysFont('Arial', 14)
        
        self.grid = np.array(MAZE)
        self.player_pos = [0, 0]
        self.mode = "manual"
        self.episodes = 0
        
        # Q-learning
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        self.epsilon = 0.9
        self.learning_rate = 0.1
        self.discount = 0.9
        
        # Training control 
        self.steps_per_second = 100
        self.min_speed = 10
        self.max_speed = 1000
        self.last_step_time = 0
        self.current_state = None
        self.episode_done = True
        
    def reset(self):
        self.player_pos = [0, 0]
        return tuple(self.player_pos)
    
    def move(self, action):
        x, y = self.player_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = moves[action]
        new_x, new_y = x + dx, y + dy
        
        if (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and 
            self.grid[new_x, new_y] != 1):
            self.player_pos = [new_x, new_y]
        
        cell = self.grid[self.player_pos[0], self.player_pos[1]]
        if cell == 3: return 100, True
        elif cell == 2: return -50, True
        else: return -1, False
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        x, y = state
        return np.argmax(self.q_table[x, y])
    
    def update_q(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        best_next = np.max(self.q_table[next_x, next_y])
        
        self.q_table[x, y, action] += self.learning_rate * (
            reward + self.discount * best_next - self.q_table[x, y, action]
        )
    
    def training_step(self):
        """Nieuwe architectuur - batch processing zoals eerste code"""
        current_time = pygame.time.get_ticks()
        
        if self.episode_done:
            self.current_state = self.reset()
            self.episode_done = False
            self.last_step_time = current_time
            return
        
        # Bereken hoeveel stappen we moeten nemen 
        delay_ms = 1000 / self.steps_per_second
        elapsed = current_time - self.last_step_time
        steps_to_take = min(50, int(elapsed / delay_ms))  # Max 50 per frame
        
        if steps_to_take > 0:
            for _ in range(steps_to_take):
                action = self.choose_action(self.current_state)
                reward, done = self.move(action)
                next_state = tuple(self.player_pos)
                
                self.update_q(self.current_state, action, reward, next_state)
                self.current_state = next_state
                
                if done:
                    self.episode_done = True
                    self.episodes += 1
                    self.epsilon = max(0.1, self.epsilon * 0.999)
                    break
            
            self.last_step_time = current_time
    
    def draw_arrow(self, surface, center, direction, strength):
        if strength <= 0:
            return
            
        arrow_size = max(8, min(20, int(strength * 2)))
        directions = [(0, -arrow_size), (arrow_size, 0), (0, arrow_size), (-arrow_size, 0)]
        
        if direction < len(directions):
            dx, dy = directions[direction]
            end_pos = (center[0] + dx, center[1] + dy)
            pygame.draw.line(surface, ARROW_COLOR, center, end_pos, 2)
            
            # Pijlpunt
            if direction == 0:
                points = [end_pos, (end_pos[0]-4, end_pos[1]+6), (end_pos[0]+4, end_pos[1]+6)]
            elif direction == 1:
                points = [end_pos, (end_pos[0]-6, end_pos[1]-4), (end_pos[0]-6, end_pos[1]+4)]
            elif direction == 2:
                points = [end_pos, (end_pos[0]-4, end_pos[1]-6), (end_pos[0]+4, end_pos[1]-6)]
            else:
                points = [end_pos, (end_pos[0]+6, end_pos[1]-4), (end_pos[0]+6, end_pos[1]+4)]
            
            pygame.draw.polygon(surface, ARROW_COLOR, points)
    
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.mode = "training"
                    self.episode_done = True  # Start fresh episode
                elif event.key == pygame.K_m:
                    self.mode = "manual"
                    self.reset()
                elif event.key == pygame.K_r:
                    self.reset()
                    self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
                    self.epsilon = 0.9
                    self.episodes = 0
                    self.episode_done = True
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.steps_per_second = min(self.max_speed, int(self.steps_per_second * 1.25))
                elif event.key == pygame.K_MINUS:
                    self.steps_per_second = max(self.min_speed, int(self.steps_per_second * 0.8))
                
                if self.mode == "manual":
                    moves = {pygame.K_UP: 0, pygame.K_RIGHT: 1, 
                            pygame.K_DOWN: 2, pygame.K_LEFT: 3}
                    if event.key in moves:
                        _, done = self.move(moves[event.key])
                        if done: self.reset()
        return True
    
    def draw(self):
        self.screen.fill(BACKGROUND)
        
        # Beide grids tekenen
        for grid_offset in [0, GRID_SIZE * CELL_SIZE + MARGIN]:
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    rect = pygame.Rect(
                        MARGIN + y * CELL_SIZE + grid_offset,
                        MARGIN + x * CELL_SIZE,
                        CELL_SIZE, CELL_SIZE
                    )
                    
                    if grid_offset == 0:  # Linker grid: maze
                        if self.grid[x, y] == 1: color = WALL_COLOR
                        elif self.grid[x, y] == 2: color = HAZARD_COLOR
                        elif self.grid[x, y] == 3: color = GOAL_COLOR
                        else: color = BACKGROUND
                        
                        pygame.draw.rect(self.screen, color, rect)
                        pygame.draw.rect(self.screen, (80, 80, 100), rect, 1)  

                        
                        if x == self.player_pos[0] and y == self.player_pos[1]:
                            pygame.draw.circle(self.screen, PLAYER_COLOR, 
                                             rect.center, CELL_SIZE//4)
                    
                    else:  # Rechter grid: Q-waarden
                        pygame.draw.rect(self.screen, (40, 40, 60), rect)
                        pygame.draw.rect(self.screen, (80, 80, 100), rect, 1)  

                        
                        if self.grid[x, y] != 1:
                            q_values = self.q_table[x, y]
                            max_q = np.max(q_values)
                            best_action = np.argmax(q_values)
                            
                            if max_q > 0:
                                self.draw_arrow(self.screen, rect.center, best_action, max_q / 10)
                                text = self.font.render(f"{max_q:.1f}", True, TEXT_COLOR)
                                self.screen.blit(text, (rect.x + 5, rect.y + 5))
        
        # UI info
        info_y = MARGIN + GRID_SIZE * CELL_SIZE + 10
        info = f"Mode: {self.mode} | Episodes: {self.episodes} | Speed: {self.steps_per_second} steps/sec | Exploration: {self.epsilon*100:.2f}%"
        text = self.font.render(info, True, TEXT_COLOR)
        self.screen.blit(text, (MARGIN, info_y))
        
        controls = "T=Train | M=Manual | R=Reset | +/-=Speed | Arrows=Move"
        controls_text = self.font.render(controls, True, TEXT_COLOR)
        self.screen.blit(controls_text, (MARGIN, info_y + 20))
        
        pygame.display.flip()
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            running = self.handle_input()
            
            if self.mode == "training":
                self.training_step()  # Nieuwe batch processing methode
            
            self.draw()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    game = SimpleMaze()
    game.run()
