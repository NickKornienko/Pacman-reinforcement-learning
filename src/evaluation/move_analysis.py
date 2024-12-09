"""
Move Analysis and Visualization
Tracks and analyzes how agents use different types of moves
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import json
from collections import defaultdict

class MoveTracker:
    """Tracks and analyzes move usage patterns"""
    def __init__(self):
        # Move categories
        self.BASIC_MOVES = ['UP', 'RIGHT', 'DOWN', 'LEFT']  # Actions 0-3
        self.DIAGONAL_MOVES = ['UP_RIGHT', 'UP_LEFT', 'DOWN_RIGHT', 'DOWN_LEFT']  # Actions 4-7
        self.SPECIAL_MOVES = ['WAIT', 'RETREAT', 'CHASE']  # Actions 8-10
        
        # Initialize counters
        self.move_counts = defaultdict(int)
        self.episode_moves = []
        self.successful_moves = defaultdict(int)  # Moves that led to reward
        self.move_sequences = []  # Track sequences of moves
    
    def track_move(self, action, reward, state=None):
        """Track a single move and its outcome"""
        # Increment move counter
        self.move_counts[action] += 1
        
        # Track successful moves (those that led to positive reward)
        if reward > 0:
            self.successful_moves[action] += 1
        
        # Store move in current episode
        self.episode_moves.append({
            'action': action,
            'reward': reward,
            'state': state
        })
    
    def end_episode(self):
        """End current episode tracking"""
        if self.episode_moves:
            self.move_sequences.append(self.episode_moves)
            self.episode_moves = []
    
    def get_move_statistics(self):
        """Calculate move usage statistics"""
        total_moves = sum(self.move_counts.values())
        if total_moves == 0:
            return {}
        
        stats = {
            'basic_usage': sum(self.move_counts[i] for i in range(4)) / total_moves,
            'diagonal_usage': sum(self.move_counts[i] for i in range(4, 8)) / total_moves,
            'special_usage': sum(self.move_counts[i] for i in range(8, 11)) / total_moves,
            'most_used': max(self.move_counts.items(), key=lambda x: x[1])[0],
            'successful_ratio': {
                action: self.successful_moves[action] / count 
                for action, count in self.move_counts.items()
                if count > 0
            }
        }
        return stats
    
    def save_stats(self, filename):
        """Save move statistics to file"""
        stats = {
            'move_counts': dict(self.move_counts),
            'successful_moves': dict(self.successful_moves),
            'statistics': self.get_move_statistics()
        }
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=4)

def create_move_usage_animation(move_sequences, save_path):
    """Create animation showing how move usage evolves over time"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.suptitle('Move Usage Evolution')
    
    frames = []
    
    # Process sequences in chunks
    chunk_size = max(1, len(move_sequences) // 30)  # 30 frames total
    for i in range(0, len(move_sequences), chunk_size):
        chunk = move_sequences[i:i+chunk_size]
        
        # Clear axes
        for ax in axes.flat:
            ax.clear()
        
        # 1. Move Type Distribution
        basic = sum(1 for seq in chunk for move in seq if move['action'] < 4)
        diagonal = sum(1 for seq in chunk for move in seq if 4 <= move['action'] < 8)
        special = sum(1 for seq in chunk for move in seq if move['action'] >= 8)
        total = basic + diagonal + special
        
        if total > 0:
            axes[0, 0].bar(['Basic', 'Diagonal', 'Special'], 
                          [basic/total, diagonal/total, special/total])
            axes[0, 0].set_title('Move Type Distribution')
            axes[0, 0].set_ylabel('Usage Ratio')
        
        # 2. Success Rate by Move Type
        success_basic = sum(1 for seq in chunk for move in seq 
                          if move['action'] < 4 and move['reward'] > 0)
        success_diagonal = sum(1 for seq in chunk for move in seq 
                             if 4 <= move['action'] < 8 and move['reward'] > 0)
        success_special = sum(1 for seq in chunk for move in seq 
                            if move['action'] >= 8 and move['reward'] > 0)
        
        if basic > 0:
            basic_rate = success_basic / basic
        else:
            basic_rate = 0
        if diagonal > 0:
            diagonal_rate = success_diagonal / diagonal
        else:
            diagonal_rate = 0
        if special > 0:
            special_rate = success_special / special
        else:
            special_rate = 0
        
        axes[0, 1].bar(['Basic', 'Diagonal', 'Special'], 
                      [basic_rate, diagonal_rate, special_rate])
        axes[0, 1].set_title('Success Rate by Move Type')
        axes[0, 1].set_ylabel('Success Rate')
        
        # 3. Move Sequence Patterns
        sequence_length = 5
        common_sequences = defaultdict(int)
        for seq in chunk:
            actions = [move['action'] for move in seq]
            for j in range(len(actions) - sequence_length + 1):
                common_sequences[tuple(actions[j:j+sequence_length])] += 1
        
        if common_sequences:
            top_sequences = sorted(common_sequences.items(), key=lambda x: x[1], reverse=True)[:5]
            axes[1, 0].bar(range(len(top_sequences)), 
                          [count for _, count in top_sequences])
            axes[1, 0].set_title('Common Move Sequences')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Reward Distribution
        rewards = [move['reward'] for seq in chunk for move in seq]
        if rewards:
            axes[1, 1].hist(rewards, bins=20)
            axes[1, 1].set_title('Reward Distribution')
            axes[1, 1].set_xlabel('Reward')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save frame
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
    
    # Save animation
    imageio.mimsave(save_path, frames, fps=2)
    plt.close()

def analyze_moves(metrics_path, gifs_path):
    """Analyze move patterns for all agents"""
    # Load metrics
    agents = ['RandomAgent', 'DQNAgent', 'DuelingDQN', 'A2C', 'MultiAgent']
    move_trackers = {agent: MoveTracker() for agent in agents}
    
    # Process metrics and create animations
    for agent in agents:
        metrics_file = os.path.join(metrics_path, f'{agent}_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Track moves
            tracker = move_trackers[agent]
            for episode in metrics.get('episodes', []):
                for step in episode.get('steps', []):
                    tracker.track_move(
                        action=step['action'],
                        reward=step['reward'],
                        state=step.get('state')
                    )
                tracker.end_episode()
            
            # Save statistics
            stats_file = os.path.join(metrics_path, f'{agent}_move_stats.json')
            tracker.save_stats(stats_file)
            
            # Create animation
            animation_file = os.path.join(gifs_path, f'{agent}_moves.gif')
            create_move_usage_animation(tracker.move_sequences, animation_file)

def main():
    # Create necessary directories
    metrics_path = '../data/metrics'
    gifs_path = '../data/gifs/move_analysis'
    os.makedirs(gifs_path, exist_ok=True)
    
    # Analyze moves and create visualizations
    analyze_moves(metrics_path, gifs_path)
    print("Move analysis complete! Check the gifs directory for animations.")

if __name__ == "__main__":
    main()
