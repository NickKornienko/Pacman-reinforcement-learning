"""
Base Pacman Environment for Reinforcement Learning
Implements core game mechanics and OpenAI Gym interface with advanced moves
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

class PacmanEnv(gym.Env):
    """
    Pacman environment that follows gym interface.
    Provides a 2D grid world where Pacman must collect dots while avoiding ghosts.
    Includes advanced movement options.
    """
    
    # Action space definitions
    BASIC_ACTIONS = {
        'UP': 0,
        'RIGHT': 1,
        'DOWN': 2,
        'LEFT': 3
    }

    DIAGONAL_ACTIONS = {
        'UP_RIGHT': 4,
        'UP_LEFT': 5,
        'DOWN_RIGHT': 6,
        'DOWN_LEFT': 7
    }

    SPECIAL_ACTIONS = {
        'WAIT': 8,
        'RETREAT': 9,
        'CHASE': 10
    }
    
    def __init__(self, grid_size=20, render_mode=None):
        super().__init__()
        
        # Environment Configuration
        self.grid_size = grid_size
        self.cell_size = 30
        self.window_size = self.grid_size * self.cell_size
        self.render_mode = render_mode
        
        # Initialize Pygame if rendering is needed
        if render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption('Pacman RL Environment')
        else:
            pygame.init()
            self.screen = pygame.Surface((self.window_size, self.window_size))
        
        # Colors
        self.COLORS = {
            'black': (0, 0, 0),
            'yellow': (255, 255, 0),
            'blue': (0, 0, 255),
            'white': (255, 255, 255),
            'red': (255, 0, 0)
        }
        
        # Action Space: Basic + Diagonal + Special actions
        self.action_space = spaces.Discrete(11)  # 4 basic + 4 diagonal + 3 special
        
        # Observation Space: grid_size x grid_size with 3 channels
        # Channel 1: Pacman position
        # Channel 2: Ghost positions
        # Channel 3: Dot positions
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, 3),
            dtype=np.float32
        )
        
        # Initialize game state
        self.reset()
    
    def _create_maze(self):
        """Create the maze layout with walls"""
        # 1 represents walls, 0 represents paths
        maze = np.ones((self.grid_size, self.grid_size))
        
        # Create paths
        for i in range(1, self.grid_size-1):
            for j in range(1, self.grid_size-1):
                if (i % 2 == 1) or (j % 2 == 1):
                    maze[i, j] = 0
        
        return maze
    
    def _get_ghost_distances(self):
        """Calculate distances to all ghosts"""
        distances = []
        for ghost in self.ghosts:
            dx = ghost[0] - self.pacman_pos[0]
            dy = ghost[1] - self.pacman_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
        return distances
    
    def _get_dot_distances(self):
        """Calculate distances to nearest dots"""
        if not self.dots:
            return [self.grid_size * 2]  # Max possible distance if no dots
        
        distances = []
        for dot in self.dots:
            dx = dot[0] - self.pacman_pos[0]
            dy = dot[1] - self.pacman_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
        return sorted(distances)[:3]  # Return distances to 3 nearest dots
    
    def _process_diagonal_action(self, action):
        """Process diagonal movement"""
        if action == self.DIAGONAL_ACTIONS['UP_RIGHT']:
            new_pos = [self.pacman_pos[0] - 1, self.pacman_pos[1] + 1]
        elif action == self.DIAGONAL_ACTIONS['UP_LEFT']:
            new_pos = [self.pacman_pos[0] - 1, self.pacman_pos[1] - 1]
        elif action == self.DIAGONAL_ACTIONS['DOWN_RIGHT']:
            new_pos = [self.pacman_pos[0] + 1, self.pacman_pos[1] + 1]
        elif action == self.DIAGONAL_ACTIONS['DOWN_LEFT']:
            new_pos = [self.pacman_pos[0] + 1, self.pacman_pos[1] - 1]
        
        # Check if both positions are valid (not wall)
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size and
            self.maze[new_pos[0], new_pos[1]] == 0):
            return new_pos
        return self.pacman_pos
    
    def _process_special_action(self, action):
        """Process special actions"""
        if action == self.SPECIAL_ACTIONS['WAIT']:
            return self.pacman_pos
        
        elif action == self.SPECIAL_ACTIONS['RETREAT']:
            # Move away from nearest ghost
            ghost_distances = self._get_ghost_distances()
            if ghost_distances:
                nearest_ghost = self.ghosts[np.argmin(ghost_distances)]
                dx = self.pacman_pos[0] - nearest_ghost[0]
                dy = self.pacman_pos[1] - nearest_ghost[1]
                new_pos = [
                    self.pacman_pos[0] + np.sign(dx),
                    self.pacman_pos[1] + np.sign(dy)
                ]
                if (0 <= new_pos[0] < self.grid_size and 
                    0 <= new_pos[1] < self.grid_size and
                    self.maze[new_pos[0], new_pos[1]] == 0):
                    return new_pos
        
        elif action == self.SPECIAL_ACTIONS['CHASE']:
            # Move toward nearest dot
            dot_distances = self._get_dot_distances()
            if dot_distances and dot_distances[0] < self.grid_size * 2:
                nearest_dot = min(self.dots, key=lambda d: 
                    np.sqrt((d[0]-self.pacman_pos[0])**2 + (d[1]-self.pacman_pos[1])**2))
                dx = nearest_dot[0] - self.pacman_pos[0]
                dy = nearest_dot[1] - self.pacman_pos[1]
                new_pos = [
                    self.pacman_pos[0] + np.sign(dx),
                    self.pacman_pos[1] + np.sign(dy)
                ]
                if (0 <= new_pos[0] < self.grid_size and 
                    0 <= new_pos[1] < self.grid_size and
                    self.maze[new_pos[0], new_pos[1]] == 0):
                    return new_pos
        
        return self.pacman_pos
    
    def reset(self, seed=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Create maze
        self.maze = self._create_maze()
        
        # Find valid positions (where maze is 0)
        valid_positions = np.argwhere(self.maze == 0)
        
        # Place Pacman randomly in valid position
        pacman_idx = self.np_random.integers(len(valid_positions))
        self.pacman_pos = valid_positions[pacman_idx].tolist()
        
        # Place ghosts in corners if valid
        self.ghosts = []
        corner_positions = [
            [1, 1], [1, self.grid_size-2],
            [self.grid_size-2, 1], [self.grid_size-2, self.grid_size-2]
        ]
        for pos in corner_positions:
            if self.maze[pos[0], pos[1]] == 0:
                self.ghosts.append(pos)
        
        # Place dots in valid positions
        self.dots = []
        for pos in valid_positions:
            pos = pos.tolist()
            if pos != self.pacman_pos and pos not in self.ghosts:
                if self.np_random.random() < 0.3:  # 30% chance of dot
                    self.dots.append(pos)
        
        # Initialize score
        self.score = 0
        
        # Get initial state
        observation = self._get_state()
        info = {"score": self.score}
        
        return observation, info
    
    def _get_state(self):
        """Convert current game state to observation space format"""
        state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Channel 1: Pacman
        state[self.pacman_pos[0], self.pacman_pos[1], 0] = 1
        
        # Channel 2: Ghosts
        for ghost in self.ghosts:
            state[ghost[0], ghost[1], 1] = 1
        
        # Channel 3: Dots
        for dot in self.dots:
            state[dot[0], dot[1], 2] = 1
        
        return state
    
    def step(self, action):
        """Take a step in the environment with advanced movement options"""
        # Store old position
        old_pos = self.pacman_pos.copy()
        
        # Process action based on type
        if action < 4:  # Basic actions
            if action == self.BASIC_ACTIONS['UP']:
                new_pos = [self.pacman_pos[0] - 1, self.pacman_pos[1]]
            elif action == self.BASIC_ACTIONS['RIGHT']:
                new_pos = [self.pacman_pos[0], self.pacman_pos[1] + 1]
            elif action == self.BASIC_ACTIONS['DOWN']:
                new_pos = [self.pacman_pos[0] + 1, self.pacman_pos[1]]
            else:  # LEFT
                new_pos = [self.pacman_pos[0], self.pacman_pos[1] - 1]
            
            # Check if new position is valid
            if (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size and
                self.maze[new_pos[0], new_pos[1]] == 0):
                self.pacman_pos = new_pos
        
        elif action < 8:  # Diagonal actions
            self.pacman_pos = self._process_diagonal_action(action)
        
        else:  # Special actions
            self.pacman_pos = self._process_special_action(action)
        
        # Calculate reward
        if self.pacman_pos in self.dots:
            self.dots.remove(self.pacman_pos)
            reward = 10
            self.score += 10
        else:
            # Higher penalty for waiting
            reward = -2 if action == self.SPECIAL_ACTIONS['WAIT'] else -1
        
        # Move ghosts randomly
        for ghost in self.ghosts:
            direction = self.np_random.integers(4)
            new_ghost_pos = ghost.copy()
            
            if direction == 0:  # UP
                new_ghost_pos[0] -= 1
            elif direction == 1:  # RIGHT
                new_ghost_pos[1] += 1
            elif direction == 2:  # DOWN
                new_ghost_pos[0] += 1
            else:  # LEFT
                new_ghost_pos[1] -= 1
            
            # Only move if new position is valid
            if (0 <= new_ghost_pos[0] < self.grid_size and
                0 <= new_ghost_pos[1] < self.grid_size and
                self.maze[new_ghost_pos[0], new_ghost_pos[1]] == 0):
                ghost[0], ghost[1] = new_ghost_pos
        
        # Check if done
        terminated = False
        
        # Ghost collision
        if self.pacman_pos in self.ghosts:
            reward = -100
            terminated = True
        
        # All dots collected
        if not self.dots:
            reward = 100
            terminated = True
        
        return self._get_state(), reward, terminated, False, {"score": self.score}
    
    def render(self):
        """Render the current state of the environment"""
        # Fill background
        self.screen.fill(self.COLORS['black'])
        
        # Draw maze
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.maze[i, j] == 1:  # Wall
                    pygame.draw.rect(
                        self.screen,
                        self.COLORS['blue'],
                        (j * self.cell_size, i * self.cell_size,
                         self.cell_size, self.cell_size)
                    )
        
        # Draw Pacman
        pygame.draw.circle(
            self.screen,
            self.COLORS['yellow'],
            (self.pacman_pos[1] * self.cell_size + self.cell_size // 2,
             self.pacman_pos[0] * self.cell_size + self.cell_size // 2),
            self.cell_size // 2 - 2
        )
        
        # Draw ghosts
        for ghost in self.ghosts:
            pygame.draw.circle(
                self.screen,
                self.COLORS['red'],
                (ghost[1] * self.cell_size + self.cell_size // 2,
                 ghost[0] * self.cell_size + self.cell_size // 2),
                self.cell_size // 2 - 2
            )
        
        # Draw dots
        for dot in self.dots:
            pygame.draw.circle(
                self.screen,
                self.COLORS['white'],
                (dot[1] * self.cell_size + self.cell_size // 2,
                 dot[0] * self.cell_size + self.cell_size // 2),
                self.cell_size // 4
            )
        
        if self.render_mode == 'human':
            pygame.display.flip()
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)),
            axes=(1, 0, 2)
        )
    
    def close(self):
        """Clean up resources"""
        pygame.quit()
