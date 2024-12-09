"""
Training script specifically for multi-agent Pacman
"""
import os
import sys
import numpy as np
from datetime import datetime
import imageio
import time
import json

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.environment.multi_agent_env import MultiAgentPacmanEnv
    from src.agents.multi_agent_system import MultiAgentSystem
except ImportError:
    print("Error: Could not import required modules.")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    raise

def save_episode_gif(frames, filename):
    """Save episode frames as GIF"""
    if frames:
        imageio.mimsave(filename, frames, fps=10)

def save_metrics(metrics, filename):
    """Save training metrics to JSON file"""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def train_multi_agent(env_config=None, train_config=None):
    """Train multi-agent system"""
    if env_config is None:
        env_config = {
            'grid_size': 20,
            'num_agents': 2,
            'render_mode': 'human',
            'cooperative': True
        }
    
    if train_config is None:
        train_config = {
            'num_episodes': 30,
            'save_gif_episodes': [0, 14, 29],
            'gif_path': os.path.join(project_root, 'data', 'gifs', 'multi_agent'),
            'model_path': os.path.join(project_root, 'data', 'models', 'multi_agent'),
            'metrics_path': os.path.join(project_root, 'data', 'metrics', 'multi_agent'),
            'step_delay': 0.1
        }
    
    # Create directories
    for path in [train_config['gif_path'], train_config['model_path'], train_config['metrics_path']]:
        os.makedirs(path, exist_ok=True)
    
    # Initialize environment
    try:
        env = MultiAgentPacmanEnv(**env_config)
    except Exception as e:
        print(f"Error initializing environment: {str(e)}")
        raise
    
    # Initialize multi-agent system
    try:
        state_shape = env.observation_space.shape
        action_size = env.action_space.n
        num_agents = env_config['num_agents']
        agent = MultiAgentSystem(state_shape, action_size, num_agents)
    except Exception as e:
        print(f"Error initializing agent system: {str(e)}")
        raise
    
    print("\nStarting Multi-Agent Training")
    print(f"Number of agents: {num_agents}")
    print(f"Mode: {'Cooperative' if env_config['cooperative'] else 'Competitive'}")
    print("\nEpisode Results:")
    print("Episode | Steps | Avg Score | Avg Reward | Max Score")
    print("-" * 60)
    
    # Training loop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        'episode_rewards': [[] for _ in range(num_agents)],
        'episode_scores': [[] for _ in range(num_agents)],
        'episode_lengths': [],
        'avg_scores': [],
        'avg_rewards': [],
        'max_scores': []
    }
    
    try:
        for episode in range(train_config['num_episodes']):
            frames = []
            save_gif = episode in train_config['save_gif_episodes']
            
            # Reset environment
            state, info = env.reset()
            episode_rewards = [0] * num_agents
            steps = 0
            
            # Episode loop
            while True:
                if save_gif:
                    frames.append(env.render())
                else:
                    env.render()
                
                time.sleep(train_config['step_delay'])
                
                # Create state list for each agent
                agent_states = [state for _ in range(num_agents)]
                
                # Get actions for all agents
                try:
                    actions = agent.select_actions(agent_states)
                except Exception as e:
                    print(f"Error selecting actions: {str(e)}")
                    print(f"State shape: {state.shape if hasattr(state, 'shape') else 'unknown'}")
                    raise
                
                # Take step in environment
                try:
                    next_state, rewards, dones, truncated, info = env.step(actions)
                except Exception as e:
                    print(f"Error stepping environment: {str(e)}")
                    print(f"Actions: {actions}")
                    raise
                
                # Store experiences for each agent
                experiences = []
                for i in range(num_agents):
                    exp = (i, state, actions[i], rewards[i], next_state, dones[i])
                    experiences.append(exp)
                agent.store_experience(experiences)
                
                # Update networks
                try:
                    agent.update_networks()
                except Exception as e:
                    print(f"Error updating networks: {str(e)}")
                    raise
                
                # Update metrics
                for i in range(num_agents):
                    episode_rewards[i] += rewards[i]
                steps += 1
                state = next_state
                
                # Check episode termination
                if any(dones) or any(truncated):
                    break
            
            # Save episode metrics
            for i in range(num_agents):
                metrics['episode_rewards'][i].append(episode_rewards[i])
                metrics['episode_scores'][i].append(info[f'agent{i}_score'])
            metrics['episode_lengths'].append(steps)
            
            # Calculate averages
            avg_score = np.mean([info[f'agent{i}_score'] for i in range(num_agents)])
            avg_reward = np.mean(episode_rewards)
            max_score = max([info[f'agent{i}_score'] for i in range(num_agents)])
            
            metrics['avg_scores'].append(avg_score)
            metrics['avg_rewards'].append(avg_reward)
            metrics['max_scores'].append(max_score)
            
            # Print episode results
            print(f"Episode {episode+1:2d} | {steps:3d} | {avg_score:6.1f} | {avg_reward:6.1f} | {max_score:6.1f}")
            
            # Save GIF if needed
            if save_gif:
                gif_filename = os.path.join(
                    train_config['gif_path'],
                    f"episode_{episode}_{timestamp}.gif"
                )
                save_episode_gif(frames, gif_filename)
            
            # Save model periodically
            if episode == train_config['num_episodes'] - 1:
                model_filename = os.path.join(
                    train_config['model_path'],
                    f"model_{timestamp}.pt"
                )
                agent.save(model_filename)
        
        # Save final metrics
        metrics_filename = os.path.join(
            train_config['metrics_path'],
            f"metrics_{timestamp}.json"
        )
        save_metrics(metrics, metrics_filename)
        
        # Print final summary
        print("\nTraining Complete!")
        print("\nFinal Performance:")
        print(f"Average Score: {np.mean(metrics['avg_scores']):.2f}")
        print(f"Average Steps: {np.mean(metrics['episode_lengths']):.2f}")
        print(f"Average Reward: {np.mean(metrics['avg_rewards']):.2f}")
        print(f"Best Episode Score: {max(metrics['max_scores']):.2f}")
        
        if env_config['cooperative']:
            print("\nTeam Performance:")
            team_scores = np.sum(metrics['episode_scores'], axis=0)
            print(f"Best Team Score: {max(team_scores):.2f}")
            print(f"Average Team Score: {np.mean(team_scores):.2f}")
        else:
            print("\nCompetitive Performance:")
            for i in range(num_agents):
                wins = sum(1 for j in range(len(metrics['episode_scores'][0])) 
                          if metrics['episode_scores'][i][j] == metrics['max_scores'][j])
                print(f"Agent {i+1} Wins: {wins}")
    
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        env.close()
    
    return metrics

if __name__ == "__main__":
    # Configure training
    env_config = {
        'grid_size': 20,
        'num_agents': 2,
        'render_mode': 'human',
        'cooperative': True
    }
    
    train_config = {
        'num_episodes': 30,
        'save_gif_episodes': [0, 14, 29],
        'gif_path': os.path.join(project_root, 'data', 'gifs', 'multi_agent'),
        'model_path': os.path.join(project_root, 'data', 'models', 'multi_agent'),
        'metrics_path': os.path.join(project_root, 'data', 'metrics', 'multi_agent'),
        'step_delay': 0.1
    }
    
    # Train multi-agent system
    try:
        metrics = train_multi_agent(env_config, train_config)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        raise
