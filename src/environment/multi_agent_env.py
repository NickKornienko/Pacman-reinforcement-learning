"""
Multi-Agent Pacman Environment
Supports multiple Pacman agents in the same environment
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

class MultiAgentPacmanEnv(gym.Env):
    """
    Multi-Agent Pacman environment where multiple agents can cooperate or compete
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
    
    def __init__(self, grid_size=20, num_agents=2, render_mode=None, cooperative=True):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.cooperative = cooperative
        self.cell_size = 30
        self.window_size = self.grid_size * self.cell_size
        
        # Initialize Pygame if rendering is needed
        if render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption('Multi-Agent Pacman')
        else:
            pygame.init()
            self.screen = pygame.Surface((self.window_size, self.window_size))
        
        # Colors
        self.COLORS = {
            'black': (0, 0, 0),
            'yellow': (255, 255, 0),
            'blue': (0, 0, 255),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'purple': (255, 0, 255)
        }
        
        # Agent colors
        self.agent_colors = [
            self.COLORS['yellow'],
            self.COLORS['green'],
            self.COLORS['purple']
        ]
        
        # Action Space: Basic + Diagonal + Special actions
        self.action_space = spaces.Discrete(11)
        
        # Observation Space: grid_size x grid_size with channels for:
        # - Each agent's position (num_agents channels)
        # - Ghost positions (1 channel)
        # - Dot positions (1 channel)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, num_agents + 2),
            dtype=np.float32
        )
    
    def _create_maze(self):
        """Create the maze layout with walls"""
        maze = np.ones((self.grid_size, self.grid_size))
        for i in range(1, self.grid_size-1):
            for j in range(1, self.grid_size-1):
                if (i % 2 == 1) or (j % 2 == 1):
                    maze[i, j] = 0
        return maze
    
    def _get_valid_positions(self, num_positions):
        """Get multiple valid positions that don't overlap"""
        valid_positions = np.argwhere(self.maze == 0)
        selected_positions = []
        
        while len(selected_positions) < num_positions and len(valid_positions) > 0:
            idx = self.np_random.integers(len(valid_positions))
            pos = valid_positions[idx].tolist()
            
            # Check if position is far enough from other selected positions
            if not any(np.sqrt((pos[0] - p[0])**2 + (pos[1] - p[1])**2) < 3 
                      for p in selected_positions):
                selected_positions.append(pos)
                valid_positions = np.delete(valid_positions, idx, axis=0)
        
        return selected_positions
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Create maze
        self.maze = self._create_maze()
        
        # Place agents
        self.agent_positions = self._get_valid_positions(self.num_agents)
        
        # Place ghosts in corners
        self.ghosts = []
        corner_positions = [
            [1, 1], [1, self.grid_size-2],
            [self.grid_size-2, 1], [self.grid_size-2, self.grid_size-2]
        ]
        for pos in corner_positions:
            if self.maze[pos[0], pos[1]] == 0:
                self.ghosts.append(pos)
        
        # Place dots
        self.dots = []
        valid_positions = np.argwhere(self.maze == 0)
        for pos in valid_positions:
            pos = pos.tolist()
            if (pos not in self.agent_positions and 
                pos not in self.ghosts and
                self.np_random.random() < 0.3):  # 30% chance of dot
                self.dots.append(pos)
        
        # Initialize scores
        self.scores = [0] * self.num_agents
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Get the current state"""
        state = np.zeros((self.grid_size, self.grid_size, self.num_agents + 2), dtype=np.float32)
        
        # Agent positions
        for i, pos in enumerate(self.agent_positions):
            state[pos[0], pos[1], i] = 1
        
        # Ghost positions
        for ghost in self.ghosts:
            state[ghost[0], ghost[1], -2] = 1
        
        # Dot positions
        for dot in self.dots:
            state[dot[0], dot[1], -1] = 1
        
        return state
    
    def _process_movement(self, agent_idx, action):
        """Process movement for a single agent"""
        old_pos = self.agent_positions[agent_idx].copy()
        new_pos = old_pos.copy()
        
        if action < 4:  # Basic actions
            if action == self.BASIC_ACTIONS['UP']:
                new_pos[0] -= 1
            elif action == self.BASIC_ACTIONS['RIGHT']:
                new_pos[1] += 1
            elif action == self.BASIC_ACTIONS['DOWN']:
                new_pos[0] += 1
            else:  # LEFT
                new_pos[1] -= 1
        
        elif action < 8:  # Diagonal actions
            if action == self.DIAGONAL_ACTIONS['UP_RIGHT']:
                new_pos[0] -= 1
                new_pos[1] += 1
            elif action == self.DIAGONAL_ACTIONS['UP_LEFT']:
                new_pos[0] -= 1
                new_pos[1] -= 1
            elif action == self.DIAGONAL_ACTIONS['DOWN_RIGHT']:
                new_pos[0] += 1
                new_pos[1] += 1
            else:  # DOWN_LEFT
                new_pos[0] += 1
                new_pos[1] -= 1
        
        elif action == self.SPECIAL_ACTIONS['WAIT']:
            return old_pos
        
        elif action == self.SPECIAL_ACTIONS['RETREAT']:
            # Move away from nearest ghost
            ghost_distances = []
            for ghost in self.ghosts:
                dx = ghost[0] - old_pos[0]
                dy = ghost[1] - old_pos[1]
                ghost_distances.append((dx, dy))
            
            if ghost_distances:
                nearest = min(ghost_distances, key=lambda d: abs(d[0]) + abs(d[1]))
                new_pos[0] = old_pos[0] - np.sign(nearest[0])
                new_pos[1] = old_pos[1] - np.sign(nearest[1])
        
        elif action == self.SPECIAL_ACTIONS['CHASE']:
            # Move toward nearest dot
            if self.dots:
                dot_distances = []
                for dot in self.dots:
                    dx = dot[0] - old_pos[0]
                    dy = dot[1] - old_pos[1]
                    dot_distances.append((dx, dy, dot))
                
                nearest = min(dot_distances, key=lambda d: abs(d[0]) + abs(d[1]))
                new_pos[0] = old_pos[0] + np.sign(nearest[0])
                new_pos[1] = old_pos[1] + np.sign(nearest[1])
        
        # Check if new position is valid
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size and
            self.maze[new_pos[0], new_pos[1]] == 0 and
            new_pos not in self.agent_positions):  # No collision with other agents
            return new_pos
        
        return old_pos
    
    def step(self, actions):
        """
        Take a step in the environment
        actions: List of actions, one for each agent
        """
        # Process movements
        new_positions = []
        for i, action in enumerate(actions):
            new_pos = self._process_movement(i, action)
            new_positions.append(new_pos)
        
        # Update positions if no conflicts
        position_counts = {}
        for pos in new_positions:
            pos_tuple = tuple(pos)
            position_counts[pos_tuple] = position_counts.get(pos_tuple, 0) + 1
        
        # Only move to positions that aren't contested
        for i, new_pos in enumerate(new_positions):
            if position_counts[tuple(new_pos)] == 1:
                self.agent_positions[i] = new_pos
        
        # Calculate rewards and update dots
        rewards = [0] * self.num_agents
        for i, pos in enumerate(self.agent_positions):
            if pos in self.dots:
                self.dots.remove(pos)
                if self.cooperative:
                    # All agents get reward in cooperative mode
                    for j in range(self.num_agents):
                        rewards[j] += 10
                        self.scores[j] += 10
                else:
                    # Only collecting agent gets reward in competitive mode
                    rewards[i] += 10
                    self.scores[i] += 10
            else:
                rewards[i] -= 1
        
        # Move ghosts
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
            
            if (0 <= new_ghost_pos[0] < self.grid_size and
                0 <= new_ghost_pos[1] < self.grid_size and
                self.maze[new_ghost_pos[0], new_ghost_pos[1]] == 0):
                ghost[0], ghost[1] = new_ghost_pos
        
        # Check for ghost collisions
        dones = [False] * self.num_agents
        for i, pos in enumerate(self.agent_positions):
            if pos in self.ghosts:
                rewards[i] -= 100
                dones[i] = True
        
        # Check if all dots collected
        if not self.dots:
            if self.cooperative:
                # All agents win in cooperative mode
                for i in range(self.num_agents):
                    rewards[i] += 100
                    dones[i] = True
            else:
                # Agent with highest score wins in competitive mode
                max_score = max(self.scores)
                for i in range(self.num_agents):
                    if self.scores[i] == max_score:
                        rewards[i] += 100
                    dones[i] = True
        
        info = {f'agent{i}_score': score for i, score in enumerate(self.scores)}
        return self._get_state(), rewards, dones, [False] * self.num_agents, info
    
    def render(self):
        """Render the environment"""
        self.screen.fill(self.COLORS['black'])
        
        # Draw maze
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.maze[i, j] == 1:
                    pygame.draw.rect(
                        self.screen,
                        self.COLORS['blue'],
                        (j * self.cell_size, i * self.cell_size,
                         self.cell_size, self.cell_size)
                    )
        
        # Draw agents
        for i, pos in enumerate(self.agent_positions):
            pygame.draw.circle(
                self.screen,
                self.agent_colors[i % len(self.agent_colors)],
                (pos[1] * self.cell_size + self.cell_size // 2,
                 pos[0] * self.cell_size + self.cell_size // 2),
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
