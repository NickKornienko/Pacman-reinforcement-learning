"""
Test script for move tracking functionality
"""
import os
import sys
import numpy as np
from datetime import datetime
import imageio

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.environment.pacman_env import PacmanEnv
from src.agents.random_agent import RandomAgent
from src.evaluation.move_analysis import create_move_usage_animation

def test_move_tracking():
    """Test move tracking with Random Agent"""
    print("Testing Move Tracking System")
    print("=" * 50)
    
    # Create directories for results
    data_dir = os.path.join(project_root, 'data')
    metrics_dir = os.path.join(data_dir, 'metrics', 'move_tracking_test')
    gifs_dir = os.path.join(data_dir, 'gifs', 'move_tracking_test')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(gifs_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = PacmanEnv(grid_size=20, render_mode='human')
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = RandomAgent(state_shape, action_size)
    
    # Run test episodes
    num_episodes = 5
    frames = []
    
    print("\nRunning Test Episodes:")
    print("Episode | Basic | Diagonal | Special | Score")
    print("-" * 45)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_frames = []
        done = False
        
        while not done:
            # Capture frame if rendering
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)
            
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Track move
            agent.track_move(action, state, next_state, reward, done or truncated)
            state = next_state
            
            if done or truncated:
                break
        
        # End episode tracking
        agent.end_episode(info['score'])
        
        # Get move distribution
        dist = agent.get_move_type_distribution()
        print(f"Episode {episode+1:2d} | {dist['basic']*100:5.1f}% | {dist['diagonal']*100:5.1f}% | "
              f"{dist['special']*100:5.1f}% | {info['score']:3d}")
        
        # Save episode GIF
        if episode_frames:
            gif_path = os.path.join(gifs_dir, f'episode_{episode+1}.gif')
            imageio.mimsave(gif_path, episode_frames, fps=10)
    
    # Save metrics
    agent.save_metrics(metrics_dir, 'RandomAgent_test')
    
    # Get final statistics
    move_stats = agent.get_move_statistics()
    
    print("\nMove Usage Statistics:")
    print("-" * 30)
    print("Move Type Distribution:")
    for move_type, pct in move_stats['move_distribution'].items():
        print(f"{move_type.capitalize():8s}: {pct*100:5.1f}%")
    
    print("\nSuccess Rates:")
    for move_type, rate in move_stats['success_rates'].items():
        print(f"{move_type.capitalize():8s}: {rate*100:5.1f}%")
    
    print("\nMost Used Moves:")
    for move_type, move_id in move_stats['most_used_moves'].items():
        move_name = agent.get_move_name(move_id)
        print(f"{move_type.capitalize():8s}: {move_name}")
    
    # Create move usage animation
    if agent.episode_moves:
        animation_path = os.path.join(gifs_dir, 'move_usage_evolution.gif')
        create_move_usage_animation(agent.episode_moves, animation_path)
        print(f"\nMove usage animation saved to: {animation_path}")
    
    print("\nTest complete! Check the data directory for results.")
    env.close()

if __name__ == "__main__":
    test_move_tracking()
