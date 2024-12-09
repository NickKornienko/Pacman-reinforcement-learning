"""
Base Agent class with move tracking functionality
"""
from collections import defaultdict
import json
import os
import numpy as np

class BaseAgent:
    """Base class for all agents with move tracking"""
    
    # Move categories
    BASIC_MOVES = {
        'UP': 0,
        'RIGHT': 1,
        'DOWN': 2,
        'LEFT': 3
    }

    DIAGONAL_MOVES = {
        'UP_RIGHT': 4,
        'UP_LEFT': 5,
        'DOWN_RIGHT': 6,
        'DOWN_LEFT': 7
    }

    SPECIAL_MOVES = {
        'WAIT': 8,
        'RETREAT': 9,
        'CHASE': 10
    }
    
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        
        # Move tracking
        self.move_counts = defaultdict(int)
        self.successful_moves = defaultdict(int)
        self.episode_moves = []
        self.episode_rewards = []
        self.episode_scores = []
        
        # Current episode tracking
        self.current_episode = []
        self.current_episode_reward = 0
        self.current_episode_score = 0
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_scores': [],
            'episode_lengths': [],
            'move_usage': defaultdict(list)  # Track move usage per episode
        }
    
    def track_move(self, action, state, next_state, reward, done):
        """Track a move and its outcome"""
        # Record move
        move_data = {
            'action': action,
            'state': state.tolist() if hasattr(state, 'tolist') else state,
            'next_state': next_state.tolist() if hasattr(next_state, 'tolist') else next_state,
            'reward': reward,
            'done': done
        }
        
        # Update counters
        self.move_counts[action] += 1
        if reward > 0:
            self.successful_moves[action] += 1
        
        # Add to current episode
        self.current_episode.append(move_data)
        self.current_episode_reward += reward
    
    def end_episode(self, final_score):
        """End current episode tracking"""
        if self.current_episode:
            # Save episode data
            self.episode_moves.append(self.current_episode)
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_scores.append(final_score)
            
            # Update training metrics
            self.training_metrics['episode_rewards'].append(self.current_episode_reward)
            self.training_metrics['episode_scores'].append(final_score)
            self.training_metrics['episode_lengths'].append(len(self.current_episode))
            
            # Track move usage for this episode
            episode_moves = defaultdict(int)
            for move in self.current_episode:
                episode_moves[move['action']] += 1
            
            for action in range(self.action_size):
                self.training_metrics['move_usage'][action].append(episode_moves[action])
            
            # Reset current episode
            self.current_episode = []
            self.current_episode_reward = 0
            self.current_episode_score = 0
    
    def get_move_statistics(self):
        """Get statistics about move usage"""
        total_moves = sum(self.move_counts.values())
        if total_moves == 0:
            return {}
        
        # Calculate move type usage
        basic_moves = sum(self.move_counts[i] for i in range(4))
        diagonal_moves = sum(self.move_counts[i] for i in range(4, 8))
        special_moves = sum(self.move_counts[i] for i in range(8, 11))
        
        # Calculate success rates
        basic_success = sum(self.successful_moves[i] for i in range(4))
        diagonal_success = sum(self.successful_moves[i] for i in range(4, 8))
        special_success = sum(self.successful_moves[i] for i in range(8, 11))
        
        stats = {
            'move_distribution': {
                'basic': basic_moves / total_moves if total_moves > 0 else 0,
                'diagonal': diagonal_moves / total_moves if total_moves > 0 else 0,
                'special': special_moves / total_moves if total_moves > 0 else 0
            },
            'success_rates': {
                'basic': basic_success / basic_moves if basic_moves > 0 else 0,
                'diagonal': diagonal_success / diagonal_moves if diagonal_moves > 0 else 0,
                'special': special_success / special_moves if special_moves > 0 else 0
            },
            'most_used_moves': {
                'basic': max(((i, self.move_counts[i]) for i in range(4)), key=lambda x: x[1])[0],
                'diagonal': max(((i, self.move_counts[i]) for i in range(4, 8)), key=lambda x: x[1])[0],
                'special': max(((i, self.move_counts[i]) for i in range(8, 11)), key=lambda x: x[1])[0]
            }
        }
        return stats
    
    def save_metrics(self, path, agent_name):
        """Save training metrics and move statistics"""
        metrics = {
            'training_metrics': self.training_metrics,
            'move_statistics': self.get_move_statistics(),
            'total_moves': dict(self.move_counts),
            'successful_moves': dict(self.successful_moves)
        }
        
        # Create metrics directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(path, f'{agent_name}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def select_action(self, state):
        """To be implemented by subclasses"""
        raise NotImplementedError
    
    def update_policy(self, *args, **kwargs):
        """To be implemented by subclasses if needed"""
        pass
    
    def get_move_name(self, action):
        """Get the name of a move from its action number"""
        for moves in [self.BASIC_MOVES, self.DIAGONAL_MOVES, self.SPECIAL_MOVES]:
            for name, value in moves.items():
                if value == action:
                    return name
        return f"Unknown_{action}"
