"""
Random Agent with move tracking
Serves as a baseline and demonstrates move tracking functionality
"""
import numpy as np
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    Agent that selects random actions.
    Used to establish baseline performance and verify move tracking.
    """
    
    def __init__(self, state_shape, action_size):
        """Initialize random agent with move tracking"""
        super().__init__(state_shape, action_size)
        
        # Additional tracking for move type analysis
        self.move_type_history = {
            'basic': [],    # Actions 0-3
            'diagonal': [], # Actions 4-7
            'special': []   # Actions 8-10
        }
    
    def select_action(self, state):
        """
        Select a random action and track its type
        
        Args:
            state: Current environment state (unused)
            
        Returns:
            action: Random action index
        """
        action = np.random.randint(self.action_size)
        
        # Track move type
        if action < 4:
            self.move_type_history['basic'].append(action)
        elif action < 8:
            self.move_type_history['diagonal'].append(action)
        else:
            self.move_type_history['special'].append(action)
        
        return action
    
    def get_move_type_distribution(self):
        """Get distribution of move types used"""
        total_moves = (len(self.move_type_history['basic']) + 
                      len(self.move_type_history['diagonal']) + 
                      len(self.move_type_history['special']))
        
        if total_moves == 0:
            return {
                'basic': 0,
                'diagonal': 0,
                'special': 0
            }
        
        return {
            'basic': len(self.move_type_history['basic']) / total_moves,
            'diagonal': len(self.move_type_history['diagonal']) / total_moves,
            'special': len(self.move_type_history['special']) / total_moves
        }
    
    def train(self, env, num_episodes=30):
        """
        Run training episodes to gather baseline performance metrics
        
        Args:
            env: Pacman environment instance
            num_episodes: Number of episodes to run
        """
        print("\nStarting Random Agent Baseline")
        print("Episode Results:")
        print("Episode | Steps | Score | Total Reward | Move Distribution")
        print("-" * 65)
        
        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                # Track move
                self.track_move(action, state, next_state, reward, done or truncated)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done or truncated:
                    break
            
            # End episode tracking
            self.end_episode(info['score'])
            
            # Get move distribution for this episode
            move_dist = self.get_move_type_distribution()
            basic_pct = move_dist['basic'] * 100
            diag_pct = move_dist['diagonal'] * 100
            special_pct = move_dist['special'] * 100
            
            # Print episode results with move distribution
            print(f"Episode {episode+1:2d} | {steps:3d} | {info['score']:3d} | {episode_reward:6.1f} | "
                  f"B:{basic_pct:3.0f}% D:{diag_pct:3.0f}% S:{special_pct:3.0f}%")
        
        # Print summary
        avg_reward = np.mean(self.episode_rewards)
        avg_score = np.mean(self.episode_scores)
        avg_length = np.mean(self.training_metrics['episode_lengths'])
        final_dist = self.get_move_type_distribution()
        
        print("\nBaseline Performance Summary:")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Average Steps: {avg_length:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print("\nOverall Move Distribution:")
        print(f"Basic Moves: {final_dist['basic']*100:.1f}%")
        print(f"Diagonal Moves: {final_dist['diagonal']*100:.1f}%")
        print(f"Special Moves: {final_dist['special']*100:.1f}%")
        
        # Get detailed move statistics
        move_stats = self.get_move_statistics()
        print("\nMove Success Rates:")
        print(f"Basic Moves: {move_stats['success_rates']['basic']*100:.1f}%")
        print(f"Diagonal Moves: {move_stats['success_rates']['diagonal']*100:.1f}%")
        print(f"Special Moves: {move_stats['success_rates']['special']*100:.1f}%")
    
    def save(self, path):
        """Save metrics and move statistics"""
        self.save_metrics(path, 'RandomAgent')
