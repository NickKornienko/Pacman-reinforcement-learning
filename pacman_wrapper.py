"""
Wrapper for PacmanEnv that adds support for sophisticated movements
"""
from pacman_env import PacmanEnv
import numpy as np
from collections import deque

class SophisticatedPacmanEnv:
    """
    Enhances PacmanEnv with:
    - Diagonal movements
    - Speed variations
    - Special actions
    - Action sequences
    """
    
    # Action definitions
    ACTIONS = {
        # Basic actions (original)
        'UP': 0,
        'RIGHT': 1,
        'DOWN': 2,
        'LEFT': 3,
        
        # Diagonal movements (combinations)
        'UP_RIGHT': 4,
        'UP_LEFT': 5,
        'DOWN_RIGHT': 6,
        'DOWN_LEFT': 7,
        
        # Speed variations (repeated actions)
        'FAST_UP': 8,
        'FAST_RIGHT': 9,
        'FAST_DOWN': 10,
        'FAST_LEFT': 11,
        
        # Special actions
        'WAIT': 12,
        'RETREAT': 13,
        'CHASE': 14
    }
    
    def __init__(self):
        self.env = PacmanEnv()
        self.action_queue = deque()
        self.last_state = None
        self.last_ghost_positions = None
        self.last_pellet_positions = None
        
    def reset(self):
        """Reset environment and clear action queue"""
        self.action_queue.clear()
        self.last_state = None
        self.last_ghost_positions = None
        self.last_pellet_positions = None
        return self.env.reset()
        
    def _get_ghost_positions(self, state):
        """Extract ghost positions from state"""
        # This is a placeholder - implement actual ghost detection
        return np.random.rand(4, 2) * 20  # Simulate 4 ghost positions
        
    def _get_pellet_positions(self, state):
        """Extract pellet positions from state"""
        # This is a placeholder - implement actual pellet detection
        return np.random.rand(10, 2) * 20  # Simulate 10 pellet positions
        
    def _process_diagonal_action(self, action):
        """Convert diagonal action to sequence of basic actions"""
        if action == self.ACTIONS['UP_RIGHT']:
            return [self.ACTIONS['UP'], self.ACTIONS['RIGHT']]
        elif action == self.ACTIONS['UP_LEFT']:
            return [self.ACTIONS['UP'], self.ACTIONS['LEFT']]
        elif action == self.ACTIONS['DOWN_RIGHT']:
            return [self.ACTIONS['DOWN'], self.ACTIONS['RIGHT']]
        elif action == self.ACTIONS['DOWN_LEFT']:
            return [self.ACTIONS['DOWN'], self.ACTIONS['LEFT']]
            
    def _process_speed_action(self, action):
        """Convert speed action to repeated basic action"""
        base_action = action - 8  # Convert to basic action
        return [base_action] * 2  # Repeat action for speed
        
    def _calculate_retreat_action(self, state):
        """Calculate best action to avoid nearest ghost"""
        ghost_positions = self._get_ghost_positions(state)
        # Simple retreat logic - move away from nearest ghost
        return self.ACTIONS['UP']  # Placeholder
        
    def _calculate_chase_action(self, state):
        """Calculate best action to reach nearest pellet"""
        pellet_positions = self._get_pellet_positions(state)
        # Simple chase logic - move toward nearest pellet
        return self.ACTIONS['RIGHT']  # Placeholder
        
    def step(self, action):
        """
        Handle sophisticated action and convert to basic actions
        Returns: next_state, reward, done, truncated, info
        """
        total_reward = 0
        final_state = None
        final_done = False
        final_truncated = False
        final_info = None
        
        try:
            # If we have queued actions, use the next one
            if self.action_queue:
                basic_action = self.action_queue.popleft()
                return self.env.step(basic_action)
                
            # Process different action types
            if action < 4:  # Basic actions
                return self.env.step(action)
                
            elif action < 8:  # Diagonal actions
                actions = self._process_diagonal_action(action)
                self.action_queue.extend(actions[1:])
                return self.env.step(actions[0])
                
            elif action < 12:  # Speed actions
                actions = self._process_speed_action(action)
                self.action_queue.extend(actions[1:])
                return self.env.step(actions[0])
                
            elif action == self.ACTIONS['WAIT']:
                # Implement wait by taking a safe action
                return self.env.step(self.ACTIONS['UP'])
                
            elif action == self.ACTIONS['RETREAT']:
                # Calculate and take retreat action
                retreat_action = self._calculate_retreat_action(self.last_state)
                return self.env.step(retreat_action)
                
            elif action == self.ACTIONS['CHASE']:
                # Calculate and take chase action
                chase_action = self._calculate_chase_action(self.last_state)
                return self.env.step(chase_action)
                
            else:
                raise ValueError(f"Invalid action: {action}")
                
        except Exception as e:
            print(f"Error in step: {e}")
            # Fallback to safe action
            return self.env.step(self.ACTIONS['UP'])
            
    def render(self):
        """Render the environment"""
        return self.env.render()
        
    def close(self):
        """Close the environment"""
        return self.env.close()
        
    def save_animation(self, filename):
        """Save the episode animation"""
        return self.env.save_animation(filename)
        
    @property
    def action_space(self):
        """Return the enhanced action space size"""
        class ActionSpace:
            def __init__(self):
                self.n = 15  # Total number of possible actions
        return ActionSpace()
