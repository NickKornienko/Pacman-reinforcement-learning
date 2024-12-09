"""
Unified training script for all Pacman RL algorithms
Handles both single-agent and multi-agent training
"""
import os
import sys
import numpy as np
from datetime import datetime
import imageio
import time
import json
import torch

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.pacman_env import PacmanEnv
from environment.multi_agent_env import MultiAgentPacmanEnv
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from agents.a2c_agent import A2CAgent
from agents.multi_agent_system import MultiAgentSystem

def save_episode_gif(frames, filename):
    """Save episode frames as GIF"""
    if frames:
        imageio.mimsave(filename, frames, fps=10)

def save_metrics(metrics, filename):
    """Save training metrics to JSON file"""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def train_single_agent(agent_class, env, train_config, agent_name):
    """Train a single agent and return metrics"""
    print(f"\nTraining {agent_name}")
    print("Episode Results:")
    print("Episode | Steps | Score | Total Reward")
    print("-" * 45)
    
    # Initialize agent
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = agent_class(state_shape, action_size)
    
    # Metrics storage
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_scores': [],
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    for episode in range(train_config['num_episodes']):
        frames = []
        save_gif = episode in train_config['save_gif_episodes']
        
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            if save_gif:
                frames.append(env.render())
            
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience (if agent supports it)
            if hasattr(agent, 'store_experience'):
                agent.store_experience(state, action, reward, next_state, done or truncated)
            
            # Update network (if agent supports it)
            if hasattr(agent, 'update_network'):
                agent.update_network()
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done or truncated:
                if save_gif:
                    frames.append(env.render())
                break
        
        # Save metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(steps)
        metrics['episode_scores'].append(info['score'])
        
        # Print episode results
        print(f"Episode {episode+1:2d} | {steps:3d} | {info['score']:3d} | {episode_reward:6.1f}")
        
        # Save GIF if needed
        if save_gif:
            gif_filename = os.path.join(
                train_config['gif_path'],
                f"{agent_name}_episode_{episode}_{metrics['timestamp']}.gif"
            )
            save_episode_gif(frames, gif_filename)
    
    # Save model
    model_filename = os.path.join(
        train_config['model_path'],
        f"{agent_name}_{metrics['timestamp']}.pt"
    )
    agent.save(model_filename)
    
    # Save metrics
    metrics_filename = os.path.join(
        train_config['metrics_path'],
        f"{agent_name}_metrics_{metrics['timestamp']}.json"
    )
    save_metrics(metrics, metrics_filename)
    
    # Print summary
    avg_reward = np.mean(metrics['episode_rewards'])
    avg_length = np.mean(metrics['episode_lengths'])
    avg_score = np.mean(metrics['episode_scores'])
    
    print(f"\n{agent_name} Training Complete!")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Steps: {avg_length:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    
    return metrics

def train_multi_agent(env, train_config):
    """Train multi-agent system"""
    print("\nTraining Multi-Agent System")
    print("Episode Results:")
    print("Episode | Steps | Avg Score | Avg Reward")
    print("-" * 55)
    
    # Initialize multi-agent system
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    num_agents = 2  # Can be configured
    agent = MultiAgentSystem(state_shape, action_size, num_agents)
    
    # Metrics storage
    metrics = {
        'episode_rewards': [[] for _ in range(num_agents)],
        'episode_lengths': [],
        'episode_scores': [[] for _ in range(num_agents)],
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    for episode in range(train_config['num_episodes']):
        frames = []
        save_gif = episode in train_config['save_gif_episodes']
        
        state, info = env.reset()
        episode_rewards = [0] * num_agents
        steps = 0
        
        while True:
            if save_gif:
                frames.append(env.render())
            
            # Select and perform actions
            actions = agent.select_actions([state] * num_agents)  # Same state for all agents
            next_state, rewards, dones, truncated, info = env.step(actions)
            
            # Update agent
            agent.store_experience([
                (i, state, action, reward, next_state, done)
                for i, (action, reward, done) in enumerate(zip(actions, rewards, dones))
            ])
            agent.update_networks()
            
            for i in range(num_agents):
                episode_rewards[i] += rewards[i]
            steps += 1
            state = next_state
            
            if any(dones) or any(truncated):
                if save_gif:
                    frames.append(env.render())
                break
        
        # Save metrics
        for i in range(num_agents):
            metrics['episode_rewards'][i].append(episode_rewards[i])
            metrics['episode_scores'][i].append(info[f'agent{i}_score'])
        metrics['episode_lengths'].append(steps)
        
        # Print episode results
        avg_score = np.mean([info[f'agent{i}_score'] for i in range(num_agents)])
        avg_reward = np.mean(episode_rewards)
        print(f"Episode {episode+1:2d} | {steps:3d} | {avg_score:6.1f} | {avg_reward:6.1f}")
        
        # Save GIF if needed
        if save_gif:
            gif_filename = os.path.join(
                train_config['gif_path'],
                f"multi_agent_episode_{episode}_{metrics['timestamp']}.gif"
            )
            save_episode_gif(frames, gif_filename)
    
    # Save model
    model_filename = os.path.join(
        train_config['model_path'],
        f"multi_agent_{metrics['timestamp']}.pt"
    )
    agent.save(model_filename)
    
    # Save metrics
    metrics_filename = os.path.join(
        train_config['metrics_path'],
        f"multi_agent_metrics_{metrics['timestamp']}.json"
    )
    save_metrics(metrics, metrics_filename)
    
    # Print summary
    avg_rewards = [np.mean(rewards) for rewards in metrics['episode_rewards']]
    avg_scores = [np.mean(scores) for scores in metrics['episode_scores']]
    avg_length = np.mean(metrics['episode_lengths'])
    
    print("\nMulti-Agent Training Complete!")
    for i in range(num_agents):
        print(f"Agent {i} - Average Score: {avg_scores[i]:.2f}, Average Reward: {avg_rewards[i]:.2f}")
    print(f"Average Steps: {avg_length:.2f}")
    
    return metrics

def train_all_agents(env_config=None, train_config=None):
    """Train all agents sequentially"""
    if env_config is None:
        env_config = {
            'grid_size': 20,
            'render_mode': 'human'
        }
    
    if train_config is None:
        train_config = {
            'num_episodes': 30,
            'save_gif_episodes': [0, 14, 29],
            'gif_path': '../data/gifs',
            'model_path': '../data/models',
            'metrics_path': '../data/metrics',
            'step_delay': 0.1
        }
    
    # Create directories
    for path in [train_config['gif_path'], train_config['model_path'], train_config['metrics_path']]:
        os.makedirs(path, exist_ok=True)
    
    # Train single agents
    single_agents = [
        (RandomAgent, "RandomAgent"),
        (DQNAgent, "DQNAgent"),
        (DuelingDQNAgent, "DuelingDQNAgent"),
        (A2CAgent, "A2CAgent")
    ]
    
    all_metrics = {}
    
    # Initialize single-agent environment
    env = PacmanEnv(**env_config)
    
    # Train single agents
    for agent_class, agent_name in single_agents:
        print(f"\n{'='*50}")
        print(f"Starting {agent_name} Training")
        print(f"{'='*50}")
        
        metrics = train_single_agent(agent_class, env, train_config, agent_name)
        all_metrics[agent_name] = metrics
        
        print(f"\n{agent_name} training completed.")
        print(f"{'='*50}\n")
        
        # Small pause between agents
        time.sleep(2)
    
    env.close()
    
    # Train multi-agent system
    print(f"\n{'='*50}")
    print("Starting Multi-Agent Training")
    print(f"{'='*50}")
    
    # Initialize multi-agent environment
    multi_env = MultiAgentPacmanEnv(**env_config)
    multi_metrics = train_multi_agent(multi_env, train_config)
    all_metrics['MultiAgent'] = multi_metrics
    
    multi_env.close()
    
    # Save combined metrics
    combined_metrics_file = os.path.join(
        train_config['metrics_path'],
        f"combined_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_metrics(all_metrics, combined_metrics_file)
    
    return all_metrics

if __name__ == "__main__":
    # Configure training
    env_config = {
        'grid_size': 20,
        'render_mode': 'human'
    }
    
    train_config = {
        'num_episodes': 30,
        'save_gif_episodes': [0, 14, 29],
        'gif_path': '../data/gifs',
        'model_path': '../data/models',
        'metrics_path': '../data/metrics',
        'step_delay': 0.1
    }
    
    # Train all agents
    metrics = train_all_agents(env_config, train_config)
