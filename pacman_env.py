"""
Basic Pacman environment with support for headless training
"""
import gymnasium as gym
import pygame
import numpy as np
from gymnasium import spaces
import imageio
import os

class PacmanEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        pygame.init()
        
        # Define constants
        self.GRID_SIZE = 20
        self.CELL_SIZE = 30
        self.WINDOW_SIZE = self.GRID_SIZE * self.CELL_SIZE
        
        # Setup display only if rendering is needed
        self.render_mode = render_mode
        if render_mode == 'human':
            self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
            pygame.display.set_caption('Pacman RL Environment')
        else:
            self.screen = pygame.Surface((self.WINDOW_SIZE, self.WINDOW_SIZE))
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        
        # Define maze layout (1 = wall, 0 = path)
        self.maze = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        
        # Action space: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 20x20 grid with 3 channels (Pacman, Ghosts, Dots)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.GRID_SIZE, self.GRID_SIZE, 3),
            dtype=np.float32
        )
        
        # Animation frames
        self.frames = []
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Clear animation frames
        self.frames = []
        
        # Find valid positions (where maze is 0)
        valid_positions = np.argwhere(self.maze == 0)
        
        # Reset Pacman to random valid position
        pacman_idx = np.random.randint(len(valid_positions))
        self.pacman_pos = valid_positions[pacman_idx].tolist()
        
        # Reset ghosts to corners (if valid)
        self.ghosts = []
        corner_positions = [[1, 1], [1, self.GRID_SIZE-2],
                            [self.GRID_SIZE-2, 1], [self.GRID_SIZE-2, self.GRID_SIZE-2]]
        for pos in corner_positions:
            if self.maze[pos[0], pos[1]] == 0:
                self.ghosts.append(pos)
        
        # Place dots in valid positions
        self.dots = []
        for pos in valid_positions:
            pos = pos.tolist()
            if pos != self.pacman_pos and pos not in self.ghosts:
                if np.random.random() < 0.3:  # 30% chance of dot
                    self.dots.append(pos)
        
        self.score = 0
        return self._get_state(), {}
    
    def _get_state(self):
        state = np.zeros((self.GRID_SIZE, self.GRID_SIZE, 3), dtype=np.float32)
        state[self.pacman_pos[0], self.pacman_pos[1], 0] = 1
        for ghost in self.ghosts:
            state[ghost[0], ghost[1], 1] = 1
        for dot in self.dots:
            state[dot[0], dot[1], 2] = 1
        return state
    
    def step(self, action):
        # Store old position
        old_pos = self.pacman_pos.copy()
        
        # Move Pacman
        if action == 0:  # UP
            new_pos = [self.pacman_pos[0] - 1, self.pacman_pos[1]]
        elif action == 1:  # RIGHT
            new_pos = [self.pacman_pos[0], self.pacman_pos[1] + 1]
        elif action == 2:  # DOWN
            new_pos = [self.pacman_pos[0] + 1, self.pacman_pos[1]]
        elif action == 3:  # LEFT
            new_pos = [self.pacman_pos[0], self.pacman_pos[1] - 1]
        
        # Check if new position is valid (not wall)
        if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
            if self.maze[new_pos[0], new_pos[1]] == 0:
                self.pacman_pos = new_pos
        
        # Check for dot collection
        if self.pacman_pos in self.dots:
            self.dots.remove(self.pacman_pos)
            reward = 10
            self.score += 10
        else:
            reward = -1
        
        # Move ghosts (only to valid positions)
        for ghost in self.ghosts:
            direction = np.random.randint(0, 4)
            new_ghost_pos = ghost.copy()
            
            if direction == 0:  # UP
                new_ghost_pos[0] -= 1
            elif direction == 1:  # RIGHT
                new_ghost_pos[1] += 1
            elif direction == 2:  # DOWN
                new_ghost_pos[0] += 1
            elif direction == 3:  # LEFT
                new_ghost_pos[1] -= 1
            
            # Only move if new position is valid
            if (0 <= new_ghost_pos[0] < self.GRID_SIZE and
                0 <= new_ghost_pos[1] < self.GRID_SIZE and
                    self.maze[new_ghost_pos[0], new_ghost_pos[1]] == 0):
                ghost[0], ghost[1] = new_ghost_pos
        
        # Check if done
        done = False
        
        # Ghost collision
        if self.pacman_pos in self.ghosts:
            reward = -100
            done = True
        
        # All dots collected
        if not self.dots:
            reward = 100
            done = True
        
        return self._get_state(), reward, done, False, {"score": self.score}
    
    def render(self):
        if self.render_mode != 'human':
            # Draw on surface
            self.screen.fill(self.BLACK)
            
            # Draw maze
            for i in range(self.GRID_SIZE):
                for j in range(self.GRID_SIZE):
                    if self.maze[i, j] == 1:  # Wall
                        pygame.draw.rect(self.screen, self.BLUE,
                                         (j * self.CELL_SIZE, i * self.CELL_SIZE,
                                          self.CELL_SIZE, self.CELL_SIZE))
            
            # Draw Pacman
            pygame.draw.circle(self.screen, self.YELLOW,
                               (self.pacman_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2,
                                self.pacman_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2),
                               self.CELL_SIZE // 2 - 2)
            
            # Draw ghosts
            for ghost in self.ghosts:
                pygame.draw.circle(self.screen, self.RED,
                                   (ghost[1] * self.CELL_SIZE + self.CELL_SIZE // 2,
                                    ghost[0] * self.CELL_SIZE + self.CELL_SIZE // 2),
                                   self.CELL_SIZE // 2 - 2)
            
            # Draw dots
            for dot in self.dots:
                pygame.draw.circle(self.screen, self.WHITE,
                                   (dot[1] * self.CELL_SIZE + self.CELL_SIZE // 2,
                                    dot[0] * self.CELL_SIZE + self.CELL_SIZE // 2),
                                   self.CELL_SIZE // 4)
            
            # Save frame for animation
            data = pygame.surfarray.array3d(self.screen)
            self.frames.append(data)
        else:
            # Draw on actual display
            self.screen.fill(self.BLACK)
            
            # Draw maze
            for i in range(self.GRID_SIZE):
                for j in range(self.GRID_SIZE):
                    if self.maze[i, j] == 1:  # Wall
                        pygame.draw.rect(self.screen, self.BLUE,
                                         (j * self.CELL_SIZE, i * self.CELL_SIZE,
                                          self.CELL_SIZE, self.CELL_SIZE))
            
            # Draw Pacman
            pygame.draw.circle(self.screen, self.YELLOW,
                               (self.pacman_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2,
                                self.pacman_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2),
                               self.CELL_SIZE // 2 - 2)
            
            # Draw ghosts
            for ghost in self.ghosts:
                pygame.draw.circle(self.screen, self.RED,
                                   (ghost[1] * self.CELL_SIZE + self.CELL_SIZE // 2,
                                    ghost[0] * self.CELL_SIZE + self.CELL_SIZE // 2),
                                   self.CELL_SIZE // 2 - 2)
            
            # Draw dots
            for dot in self.dots:
                pygame.draw.circle(self.screen, self.WHITE,
                                   (dot[1] * self.CELL_SIZE + self.CELL_SIZE // 2,
                                    dot[0] * self.CELL_SIZE + self.CELL_SIZE // 2),
                                   self.CELL_SIZE // 4)
            
            pygame.display.flip()
            
            # Save frame for animation
            data = pygame.surfarray.array3d(self.screen)
            self.frames.append(data)
    
    def save_animation(self, filename='pacman_animation.gif'):
        if self.frames:
            # Ensure frames are in correct format
            frames = [np.transpose(frame, (1, 0, 2)) for frame in self.frames]
            # Save as GIF
            imageio.mimsave(filename, frames, fps=10)
            print(f"Animation saved as {filename}")
            # Clear frames
            self.frames = []
    
    def close(self):
        pygame.quit()
