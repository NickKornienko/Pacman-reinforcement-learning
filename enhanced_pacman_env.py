"""
Enhanced Pacman Environment that adds sophisticated action handling
while maintaining compatibility with the base environment
"""
from pacman_env import PacmanEnv
import numpy as np
from collections import deque
import time


class EnhancedPacmanEnv:
    """
    Wrapper around PacmanEnv that adds:
    - Diagonal movements
    - Variable speed movements
    - Special actions (wait, retreat, chase)
    - Action sequences
    - State history for temporal features
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

    SPEED_ACTIONS = {
        'FAST_UP': 8,
        'FAST_RIGHT': 9,
        'FAST_DOWN': 10,
        'FAST_LEFT': 11
    }

    SPECIAL_ACTIONS = {
        'WAIT': 12,
        'RETREAT': 13,
        'CHASE': 14
    }

    def __init__(self):
        self.env = PacmanEnv()
        self.action_space_size = 15
        # Keep last 4 states for temporal features
        self.state_history = deque(maxlen=4)
        self.last_ghost_distances = None
        self.last_pellet_distances = None
        self.action_cooldown = 0

    def reset(self):
        obs, info = self.env.reset()
        self.state_history.clear()
        for _ in range(4):
            self.state_history.append(obs)
        self.last_ghost_distances = self._get_ghost_distances(obs)
        self.last_pellet_distances = self._get_pellet_distances(obs)
        self.action_cooldown = 0
        return self._get_enhanced_state(obs), info

    def _get_ghost_distances(self, state):
        """Calculate distances to ghosts"""
        # This is a placeholder - implement actual ghost detection
        return np.random.rand(4) * 20  # Simulate 4 ghost distances

    def _get_pellet_distances(self, state):
        """Calculate distances to nearest pellets"""
        # This is a placeholder - implement actual pellet detection
        return np.random.rand(4) * 20  # Simulate 4 pellet distances

    def _get_enhanced_state(self, obs):
        """Combine current observation with temporal and distance features"""
        ghost_distances = self._get_ghost_distances(obs)
        pellet_distances = self._get_pellet_distances(obs)

        # Calculate ghost movement vectors
        ghost_movements = ghost_distances - \
            self.last_ghost_distances if self.last_ghost_distances is not None else np.zeros(
                4)

        # Calculate pellet changes
        pellet_changes = pellet_distances - \
            self.last_pellet_distances if self.last_pellet_distances is not None else np.zeros(
                4)

        self.last_ghost_distances = ghost_distances
        self.last_pellet_distances = pellet_distances

        # Return enhanced state
        return {
            'pixels': obs,
            'ghost_distances': ghost_distances,
            'pellet_distances': pellet_distances,
            'ghost_movements': ghost_movements,
            'pellet_changes': pellet_changes,
            'history': list(self.state_history)
        }

    def _process_diagonal_action(self, action):
        """Convert diagonal action to sequence of basic actions"""
        if action == self.DIAGONAL_ACTIONS['UP_RIGHT']:
            return [self.BASIC_ACTIONS['UP'], self.BASIC_ACTIONS['RIGHT']]
        elif action == self.DIAGONAL_ACTIONS['UP_LEFT']:
            return [self.BASIC_ACTIONS['UP'], self.BASIC_ACTIONS['LEFT']]
        elif action == self.DIAGONAL_ACTIONS['DOWN_RIGHT']:
            return [self.BASIC_ACTIONS['DOWN'], self.BASIC_ACTIONS['RIGHT']]
        elif action == self.DIAGONAL_ACTIONS['DOWN_LEFT']:
            return [self.BASIC_ACTIONS['DOWN'], self.BASIC_ACTIONS['LEFT']]

    def _process_speed_action(self, action):
        """Convert speed action to repeated basic actions"""
        base_action = action - 8  # Convert to basic action
        return [base_action] * 2  # Repeat action for speed

    def _process_special_action(self, action, state):
        """Handle special actions like WAIT, RETREAT, CHASE"""
        if action == self.SPECIAL_ACTIONS['WAIT']:
            self.action_cooldown = 2  # Wait for 2 steps
            return None

        elif action == self.SPECIAL_ACTIONS['RETREAT']:
            # Find safest direction based on ghost distances
            ghost_distances = state['ghost_distances']
            safest_direction = np.argmax(ghost_distances)
            return safest_direction

        elif action == self.SPECIAL_ACTIONS['CHASE']:
            # Find best direction based on pellet distances
            pellet_distances = state['pellet_distances']
            best_direction = np.argmin(pellet_distances)
            return best_direction

    def step(self, action):
        """
        Handle enhanced action space by converting to basic actions
        Returns: next_state, reward, done, truncated, info
        """
        total_reward = 0
        done = False
        truncated = False
        info = None
        current_state = None

        # Handle action cooldown
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
            current_state, reward, done, truncated, info = self.env.step(
                self.BASIC_ACTIONS['WAIT'])
            return self._get_enhanced_state(current_state), reward, done, truncated, info

        # Process different action types
        if action < 4:  # Basic actions
            current_state, reward, done, truncated, info = self.env.step(
                action)
            total_reward += reward

        elif action < 8:  # Diagonal actions
            action_sequence = self._process_diagonal_action(action)
            for sub_action in action_sequence:
                if not done:
                    current_state, reward, done, truncated, info = self.env.step(
                        sub_action)
                    total_reward += reward

        elif action < 12:  # Speed actions
            action_sequence = self._process_speed_action(action)
            for sub_action in action_sequence:
                if not done:
                    current_state, reward, done, truncated, info = self.env.step(
                        sub_action)
                    total_reward += reward

        else:  # Special actions
            enhanced_state = self._get_enhanced_state(self.state_history[-1])
            processed_action = self._process_special_action(
                action, enhanced_state)
            if processed_action is not None:
                current_state, reward, done, truncated, info = self.env.step(
                    processed_action)
                total_reward += reward
            else:
                current_state = self.state_history[-1]
                reward = 0

        # Update state history
        self.state_history.append(current_state)

        # Return enhanced state
        return self._get_enhanced_state(current_state), total_reward, done, truncated, info

    def render(self):
        """Render the environment"""
        return self.env.render()

    def close(self):
        """Close the environment"""
        return self.env.close()

    def save_animation(self, filename):
        """Save the episode animation"""
        return self.env.save_animation(filename)
