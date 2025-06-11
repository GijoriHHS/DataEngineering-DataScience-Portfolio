import pygame
import numpy as np
import random
import os

# Constanten
GRID_SIZE = 6
CELL_SIZE = 125
MARGIN = 30
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE * 2 + MARGIN * 3
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + MARGIN * 2 + 80

# Kleuren
BACKGROUND = (30, 30, 50)
WALL_COLOR = (85, 85, 98)
GOAL_COLOR = (50, 180, 80)
HAZARD_COLOR = (220, 80, 60)
TEXT_COLOR = (220, 220, 220)
Q_VALUE_COLOR = (255, 255, 100)

# Maze (0=pad, 1=muur, 2=gevaar, 3=doel)
MAZE = [
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 2, 0],
    [0, 0, 0, 0, 1, 3],
    [0, 1, 1, 0, 1, 0],
    [0, 2, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0]
]

DIRECTIONS = ['Up', 'Right', 'Down', 'Left']

class PlayerSprite:
    def __init__(self, sprite_folder, cell_size):
        self.images = {}
        for dir_name in DIRECTIONS:
            path = os.path.join(sprite_folder, f"{dir_name}.png")
            img = pygame.image.load(path).convert_alpha()
            self.images[dir_name] = pygame.transform.smoothscale(img, (cell_size-10, cell_size-10))
        self.direction = 'Down'

    def set_direction(self, action):
        self.direction = DIRECTIONS[action]

    def draw(self, surface, center):
        img = self.images[self.direction]
        rect = img.get_rect(center=center)
        surface.blit(img, rect)

class TileSprites:
    def __init__(self, sprite_folder, cell_size):
        finish_path = os.path.join(sprite_folder, "Finish.png")
        hazard_path = os.path.join(sprite_folder, "Hazard.png")
        self.finish = pygame.image.load(finish_path).convert_alpha()
        self.finish = pygame.transform.smoothscale(self.finish, (cell_size-8, cell_size-8))
        self.hazard = pygame.image.load(hazard_path).convert_alpha()
        self.hazard = pygame.transform.smoothscale(self.hazard, (cell_size-8, cell_size-8))

class SimpleMaze:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Q-Learning Maze")
        self.font = pygame.font.SysFont('Arial', 10)  # Kleinere font voor 4 waarden
        
        self.grid = np.array(MAZE)
        self.player_pos = [0, 0]
        self.mode = "manual"
        self.episodes = 0

        try:
            sprite_folder = os.path.join(os.path.dirname(__file__), 'Sprites')
        except NameError:
            sprite_folder = os.path.join(os.getcwd(), 'Sprites')
            
        if not os.path.exists(sprite_folder):
            print(f"Sprites map niet gevonden op: {sprite_folder}")
            print("Zorg dat de Sprites map in dezelfde directory staat als je notebook")

        self.player_sprite = PlayerSprite(sprite_folder, CELL_SIZE)
        self.tile_sprites = TileSprites(sprite_folder, CELL_SIZE)
        self.last_action = 2  # DOWN
        
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
        self.last_action = 2
        self.player_sprite.set_direction(self.last_action)
        return tuple(self.player_pos)
    
    def move(self, action):
        x, y = self.player_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = moves[action]
        new_x, new_y = x + dx, y + dy
        
        if (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and 
            self.grid[new_x, new_y] != 1):
            self.player_pos = [new_x, new_y]
            self.last_action = action
            self.player_sprite.set_direction(action)
        
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
        current_time = pygame.time.get_ticks()
        
        if self.episode_done:
            self.current_state = self.reset()
            self.episode_done = False
            self.last_step_time = current_time
            return
        
        delay_ms = 1000 / self.steps_per_second
        elapsed = current_time - self.last_step_time
        steps_to_take = min(50, int(elapsed / delay_ms))
        
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
    
    def draw_q_values(self, surface, rect, q_values):
        """Tekent alle 4 Q-waarden in de juiste posities binnen een tile"""
        # Posities: Up, Right, Down, Left
        positions = [
            (rect.centerx, rect.y + 8),      # Up - boven in het midden
            (rect.right - 25, rect.centery), # Right - rechts in het midden  
            (rect.centerx, rect.bottom - 15), # Down - beneden in het midden
            (rect.x + 8, rect.centery)       # Left - links in het midden
        ]
        
        # Kleur bepalen op basis van waarde (groen voor positief, rood voor negatief)
        for i, (q_value, pos) in enumerate(zip(q_values, positions)):
            if abs(q_value) > 0.01:  # Alleen tonen als waarde significant is
                # Kleur bepalen
                if q_value > 0:
                    color = (100, 255, 100)  # Groen voor positieve waarden
                elif q_value < 0:
                    color = (255, 100, 100)  # Rood voor negatieve waarden
                else:
                    color = Q_VALUE_COLOR     # Geel voor neutrale waarden
                
                # Tekst renderen
                text = self.font.render(f"{q_value:.1f}", True, color)
                text_rect = text.get_rect()
                
                # Centreren op de positie
                text_rect.center = pos
                
                # Zorgen dat tekst binnen de tile blijft
                text_rect.clamp_ip(rect.inflate(-4, -4))
                
                surface.blit(text, text_rect)
    
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.mode = "training"
                    self.episode_done = True
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
                        # Eerst de juiste achtergrondkleur tekenen
                        if self.grid[x, y] == 1:
                            pygame.draw.rect(self.screen, WALL_COLOR, rect)
                        elif self.grid[x, y] == 2:
                            pygame.draw.rect(self.screen, HAZARD_COLOR, rect)  
                            self.screen.blit(self.tile_sprites.hazard, rect.inflate(-8, -8))
                        elif self.grid[x, y] == 3:
                            pygame.draw.rect(self.screen, GOAL_COLOR, rect)  
                            self.screen.blit(self.tile_sprites.finish, rect.inflate(-8, -8))
                        else:
                            pygame.draw.rect(self.screen, BACKGROUND, rect)
                        
                        pygame.draw.rect(self.screen, (110, 110, 130), rect, 1)  

                        # Player tekenen als sprite
                        if x == self.player_pos[0] and y == self.player_pos[1]:
                            self.player_sprite.draw(self.screen, rect.center)
                    
                    else:  # Rechter grid: Q-waarden
                        pygame.draw.rect(self.screen, (40, 40, 60), rect)
                        pygame.draw.rect(self.screen, (110, 110, 130), rect, 1)   

                        if self.grid[x, y] != 1:  # Geen Q-waarden voor muren
                            q_values = self.q_table[x, y]
                            self.draw_q_values(self.screen, rect, q_values)
        
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
                self.training_step()
            
            self.draw()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    game = SimpleMaze()
    game.run()
