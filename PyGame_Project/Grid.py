import pygame
import numpy as np
import random
import time

# Initialize PyGame
pygame.init()

class GridWorld:
    def __init__(self, size=5):
        self.size = size  # Grid size (e.g., 5x5)
        self.grid = self.initialize_grid()  # Create the grid
        self.agent_pos = [0, 0]  # Starting position (top-left corner)
        self.goal_pos = [size-1, size-1]  # Goal at bottom-right
        self.actions = ['up', 'down', 'left', 'right']  # Possible actions

    def initialize_grid(self):
        # Create a 5x5 grid filled with zeros (empty spaces)
        grid = np.zeros((self.size, self.size), dtype=int)
        # Add walls (example positions, you can customize)
        grid[1, 1] = 1
        grid[2, 2] = 1
        grid[3, 1] = 1
        # Add hazard
        grid[2, 1] = 2
        # Add goal
        grid[self.size-1, self.size-1] = 3
        return grid

    def reset(self):
        # Reset agent to starting position
        self.agent_pos = [0, 0]
        return self.agent_pos

    def is_valid_move(self, pos):
        row, col = pos
        # Check if move is within grid bounds
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        # Check if move is into a wall
        if self.grid[row, col] == 1:
            return False
        return True

    def move_agent(self, action):
        # Calculate new position based on action
        new_pos = self.agent_pos.copy()
        if action == 'up':
            new_pos[0] -= 1
        elif action == 'down':
            new_pos[0] += 1
        elif action == 'left':
            new_pos[1] -= 1
        elif action == 'right':
            new_pos[1] += 1

        # Check if the move is valid
        if self.is_valid_move(new_pos):
            self.agent_pos = new_pos
        # Return current position, reward, and whether goal is reached
        reward = self.calculate_reward(self.agent_pos)
        done = self.grid[self.agent_pos[0], self.agent_pos[1]] == 3
        return self.agent_pos, reward, done

    def calculate_reward(self, pos):
        row, col = pos
        if self.grid[row, col] == 2:  # Hazard
            return -10
        elif self.grid[row, col] == 3:  # Goal
            return 100
        return -1  # Small penalty for each step to encourage efficiency    
    
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.env = env  # Reference to the GridWorld environment
        self.lr = learning_rate  # Learning rate (alpha)
        self.gamma = discount_factor  # Discount factor (gamma)
        self.epsilon = epsilon  # Exploration rate for epsilon-greedy
        self.q_table = np.zeros((env.size, env.size, len(env.actions)))  # Q-table: size x size x actions
        self.actions = env.actions  # List of actions

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            return random.choice(self.actions)
        else:
            # Exploitation: choose the action with the highest Q-value
            row, col = state
            return self.actions[np.argmax(self.q_table[row, col])]

    def update_q_value(self, state, action, reward, next_state):
        # Convert action to index
        action_idx = self.actions.index(action)
        row, col = state
        next_row, next_col = next_state
        # Q-Learning update rule
        current_q = self.q_table[row, col, action_idx]
        max_next_q = np.max(self.q_table[next_row, next_col])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[row, col, action_idx] = new_q

class GameRenderer:
    def __init__(self, env, agent, cell_size=50):
        self.env = env
        self.agent = agent
        self.cell_size = cell_size
        self.screen_width = env.size * cell_size * 2  # Double for Q-value grid
        self.screen_height = env.size * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Q-Learning Grid World")
        self.font = pygame.font.SysFont('arial', 12)
        self.colors = {
            0: (200, 200, 200),  # Empty: Gray
            1: (0, 0, 0),       # Wall: Black
            2: (255, 0, 0),     # Hazard: Red
            3: (0, 255, 0)      # Goal: Green
        }
        self.agent_color = (0, 0, 255)  # Agent: Blue

    def render(self):
        self.screen.fill((255, 255, 255))  # White background

        # Draw the grid world
        for row in range(self.env.size):
            for col in range(self.env.size):
                cell_type = self.env.grid[row, col]
                pygame.draw.rect(self.screen, self.colors[cell_type],
                                (col * self.cell_size, row * self.cell_size,
                                 self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (0, 0, 0),
                                (col * self.cell_size, row * self.cell_size,
                                 self.cell_size, self.cell_size), 1)

        # Draw the agent
        agent_row, agent_col = self.env.agent_pos
        pygame.draw.circle(self.screen, self.agent_color,
                          (agent_col * self.cell_size + self.cell_size // 2,
                           agent_row * self.cell_size + self.cell_size // 2),
                           self.cell_size // 3)

        # Draw Q-values
        for row in range(self.env.size):
            for col in range(self.env.size):
                for action_idx, action in enumerate(self.agent.actions):
                    q_value = self.agent.q_table[row, col, action_idx]
                    text = self.font.render(f"{q_value:.1f}", True, (0, 0, 0))
                    x = (self.env.size + col) * self.cell_size
                    y = row * self.cell_size + (action_idx + 1) * (self.cell_size // 5)
                    self.screen.blit(text, (x, y))

        pygame.display.flip()

def main():
    env = GridWorld(size=5)
    agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
    renderer = GameRenderer(env, agent)
    running = True
    manual_mode = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Start training
                    manual_mode = False
                    train_agent(env, agent, renderer, episodes=1000)
                    manual_mode = True
                elif event.key == pygame.K_r:  # Reset environment
                    env.reset()
                elif manual_mode and event.type == pygame.KEYDOWN:
                    action = None
                    if event.key == pygame.K_UP:
                        action = 'up'
                    elif event.key == pygame.K_DOWN:
                        action = 'down'
                    elif event.key == pygame.K_LEFT:
                        action = 'left'
                    elif event.key == pygame.K_RIGHT:
                        action = 'right'
                    if action:
                        env.move_agent(action)
                        renderer.render()

        if manual_mode:
            renderer.render()

    pygame.quit()

def train_agent(env, agent, renderer, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.move_agent(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            renderer.render()
            time.sleep(0.05)  # Slow down to visualize
        print(f"Episode {episode + 1}/{episodes} completed")

if __name__ == "__main__":
    main()