"""
Comprehensive comparison of all algorithms with flexible training/evaluation
"""
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import imageio
import time
import argparse
from collections import defaultdict

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from environment.pacman_env import PacmanEnv
from environment.multi_agent_env import MultiAgentPacmanEnv
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from agents.a2c_agent import A2CAgent
from agents.multi_agent_system import MultiAgentSystem

def save_episode_gif(frames, path, fps=30):
    """Save frames as animated GIF"""
    if frames:
        frames = [np.array(frame).astype(np.uint8) for frame in frames]
        imageio.mimsave(path, frames, fps=fps)
        print(f"Saved animated GIF: {path}")

def train_single_agent(agent_class, env, num_episodes, dirs, name):
    """Train and evaluate a single agent"""
    print(f"\nTraining {name}")
    print("=" * 50)
    
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = agent_class(state_shape, action_size)
    
    metrics = defaultdict(list)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        episode_frames = []
        
        episode_start = time.time()
        
        while True:
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)
            
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            if hasattr(agent, 'store_experience'):
                agent.store_experience(state, action, reward, next_state, done)
            
            if hasattr(agent, 'update_network'):
                agent.update_network()
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done or truncated:
                break
        
        episode_duration = time.time() - episode_start
        
        if episode in [0, num_episodes//2, num_episodes-1]:
            gif_path = os.path.join(dirs['gifs'], f'{name}_episode_{episode}.gif')
            save_episode_gif(episode_frames, gif_path)
        
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_scores'].append(info['score'])
        metrics['episode_lengths'].append(steps)
        metrics['episode_durations'].append(episode_duration)
        
        print(f"Episode {episode+1}/{num_episodes} - Steps: {steps}, Score: {info['score']}, "
              f"Reward: {episode_reward:.2f}, Duration: {episode_duration:.2f}s")
    
    # Save model
    model_path = os.path.join(dirs['models'], f'{name}_final.pt')
    agent.save(model_path)
    
    # Save metrics
    metrics_path = os.path.join(dirs['metrics'], f'{name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return agent, metrics

def train_multi_agent(env, num_episodes, dirs):
    """Train and evaluate multi-agent system"""
    print("\nTraining Multi-Agent System")
    print("=" * 50)
    
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    num_agents = 2
    agent = MultiAgentSystem(state_shape, action_size, num_agents)
    
    metrics = defaultdict(list)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = [0] * num_agents
        steps = 0
        episode_frames = []
        
        episode_start = time.time()
        
        while True:
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)
            
            actions = agent.select_actions([state] * num_agents)
            next_state, rewards, dones, truncated, info = env.step(actions)
            
            experiences = [
                (i, state, action, reward, next_state, done)
                for i, (action, reward, done) in enumerate(zip(actions, rewards, dones))
            ]
            agent.store_experience(experiences)
            agent.update_networks()
            
            for i in range(num_agents):
                episode_reward[i] += rewards[i]
            steps += 1
            state = next_state
            
            if any(dones) or any(truncated):
                break
        
        episode_duration = time.time() - episode_start
        
        if episode in [0, num_episodes//2, num_episodes-1]:
            gif_path = os.path.join(dirs['gifs'], f'multi_agent_episode_{episode}.gif')
            save_episode_gif(episode_frames, gif_path)
        
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_scores'].append([info[f'agent{i}_score'] for i in range(num_agents)])
        metrics['episode_lengths'].append(steps)
        metrics['episode_durations'].append(episode_duration)
        
        avg_reward = np.mean(episode_reward)
        print(f"Episode {episode+1}/{num_episodes} - Steps: {steps}, "
              f"Average Reward: {avg_reward:.2f}, Duration: {episode_duration:.2f}s")
    
    # Save model
    model_path = os.path.join(dirs['models'], 'multi_agent_final.pt')
    agent.save(model_path)
    
    # Save metrics
    metrics_path = os.path.join(dirs['metrics'], 'multi_agent_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return agent, metrics

def evaluate_saved_models(dirs, agents_to_evaluate=None):
    """Evaluate saved models without retraining"""
    print("\nEvaluating Saved Models")
    print("=" * 50)
    
    results = {}
    env = PacmanEnv(grid_size=20, render_mode='rgb_array')
    
    # Define available agents
    agent_classes = {
        'random': RandomAgent,
        'dqn': DQNAgent,
        'dueling_dqn': DuelingDQNAgent,
        'a2c': A2CAgent
    }
    
    # Filter agents if specified
    if agents_to_evaluate:
        agent_classes = {name: cls for name, cls in agent_classes.items() 
                        if name in agents_to_evaluate}
    
    # Evaluate each agent
    for name, agent_class in agent_classes.items():
        print(f"\nEvaluating {name}")
        
        # Initialize agent
        agent = agent_class(env.observation_space.shape, env.action_space.n)
        
        # Load model if it's not the random agent
        if name != 'random':
            model_path = os.path.join(dirs['models'], f'{name}_final.pt')
            if not os.path.exists(model_path):
                print(f"Model not found for {name}, skipping...")
                continue
            agent.load(model_path)
        
        metrics = defaultdict(list)
        num_eval_episodes = 5
        
        for episode in range(num_eval_episodes):
            state, info = env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done or truncated:
                    break
            
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_scores'].append(info['score'])
            metrics['episode_lengths'].append(steps)
            
            print(f"Episode {episode+1}/{num_eval_episodes} - "
                  f"Score: {info['score']}, Reward: {episode_reward:.2f}")
        
        results[name] = metrics
    
    env.close()
    
    # Evaluate multi-agent system if available
    multi_model_path = os.path.join(dirs['models'], 'multi_agent_final.pt')
    if os.path.exists(multi_model_path) and (not agents_to_evaluate or 'multi_agent' in agents_to_evaluate):
        print("\nEvaluating Multi-Agent System")
        multi_env = MultiAgentPacmanEnv(grid_size=20, render_mode='rgb_array')
        
        agent = MultiAgentSystem(multi_env.observation_space.shape, 
                               multi_env.action_space.n, 
                               num_agents=2)
        agent.load(multi_model_path)
        
        metrics = defaultdict(list)
        num_eval_episodes = 5
        
        for episode in range(num_eval_episodes):
            state, info = multi_env.reset()
            episode_reward = [0, 0]
            steps = 0
            
            while True:
                actions = agent.select_actions([state] * 2)
                next_state, rewards, dones, truncated, info = multi_env.step(actions)
                
                for i in range(2):
                    episode_reward[i] += rewards[i]
                steps += 1
                state = next_state
                
                if any(dones) or any(truncated):
                    break
            
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_scores'].append([info[f'agent{i}_score'] for i in range(2)])
            metrics['episode_lengths'].append(steps)
            
            team_score = sum(info[f'agent{i}_score'] for i in range(2))
            print(f"Episode {episode+1}/{num_eval_episodes} - Team Score: {team_score}")
        
        results['multi_agent'] = metrics
        multi_env.close()
    
    # Save evaluation results
    eval_path = os.path.join(dirs['reports'], 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def generate_comparison_plots(results, dirs):
    """Generate comparison plots from results"""
    # 1. Score Comparison
    plt.figure(figsize=(12, 6))
    for agent_name, metrics in results.items():
        if agent_name != 'multi_agent':
            scores = metrics['episode_scores']
            plt.plot(scores, label=agent_name)
    plt.title('Score Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(os.path.join(dirs['plots'], 'score_comparison.png'))
    plt.close()
    
    # 2. Multi-Agent Performance (if available)
    if 'multi_agent' in results:
        plt.figure(figsize=(12, 6))
        multi_metrics = results['multi_agent']
        episodes = range(len(multi_metrics['episode_scores']))
        
        for i in range(2):
            agent_scores = [scores[i] for scores in multi_metrics['episode_scores']]
            plt.plot(episodes, agent_scores, label=f'Agent {i+1}')
        
        team_scores = [sum(scores) for scores in multi_metrics['episode_scores']]
        plt.plot(episodes, team_scores, label='Team Total', linestyle='--')
        
        plt.title('Multi-Agent Performance')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(os.path.join(dirs['plots'], 'multi_agent_performance.png'))
        plt.close()
    
    # 3. Generate summary report
    report = ["# Algorithm Comparison Report\n"]
    report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for agent_name, metrics in results.items():
        report.append(f"## {agent_name}")
        
        if agent_name == 'multi_agent':
            team_scores = [sum(scores) for scores in metrics['episode_scores']]
            report.extend([
                f"- Average Team Score: {np.mean(team_scores):.2f}",
                f"- Best Team Score: {max(team_scores):.2f}",
                f"- Average Episode Length: {np.mean(metrics['episode_lengths']):.2f}\n"
            ])
            
            for i in range(2):
                agent_scores = [scores[i] for scores in metrics['episode_scores']]
                report.extend([
                    f"### Agent {i+1}",
                    f"- Average Score: {np.mean(agent_scores):.2f}",
                    f"- Best Score: {max(agent_scores):.2f}\n"
                ])
        else:
            report.extend([
                f"- Average Score: {np.mean(metrics['episode_scores']):.2f}",
                f"- Best Score: {max(metrics['episode_scores']):.2f}",
                f"- Average Episode Length: {np.mean(metrics['episode_lengths']):.2f}\n"
            ])
    
    report_path = os.path.join(dirs['reports'], 'comparison_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate RL agents')
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train',
                      help='Mode of operation')
    parser.add_argument('--agents', nargs='+', 
                      help='Specific agents to train/evaluate (e.g., random dqn multi_agent)')
    parser.add_argument('--data-dir', default='data',
                      help='Directory containing saved models and metrics')
    
    args = parser.parse_args()
    
    # Setup directories
    dirs = {
        'base': args.data_dir,
        'models': os.path.join(args.data_dir, 'models'),
        'gifs': os.path.join(args.data_dir, 'gifs'),
        'plots': os.path.join(args.data_dir, 'plots'),
        'metrics': os.path.join(args.data_dir, 'metrics'),
        'reports': os.path.join(args.data_dir, 'reports')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Available agents
    available_agents = {
        'random': (RandomAgent, 'random'),
        'dqn': (DQNAgent, 'dqn'),
        'dueling_dqn': (DuelingDQNAgent, 'dueling_dqn'),
        'a2c': (A2CAgent, 'a2c'),
        'multi_agent': None  # Handled separately
    }
    
    if args.mode == 'train':
        print("Starting Training Mode")
        
        # Filter agents if specified
        agents_to_train = available_agents
        if args.agents:
            agents_to_train = {name: agent for name, agent in available_agents.items() 
                             if name in args.agents}
        
        results = {}
        
        # Train single agents
        if any(name != 'multi_agent' for name in agents_to_train.keys()):
            env = PacmanEnv(grid_size=20, render_mode='rgb_array')
            
            for name, (agent_class, save_name) in agents_to_train.items():
                if name != 'multi_agent':
                    print(f"\nTraining {name}")
                    agent, metrics = train_single_agent(
                        agent_class, env, 30, dirs, save_name)
                    results[name] = metrics
            
            env.close()
        
        # Train multi-agent system if specified
        if 'multi_agent' in agents_to_train:
            print("\nTraining Multi-Agent System")
            multi_env = MultiAgentPacmanEnv(grid_size=20, render_mode='rgb_array')
            multi_agent, multi_metrics = train_multi_agent(multi_env, 30, dirs)
            results['multi_agent'] = multi_metrics
            multi_env.close()
        
        # Generate comparison plots
        if results:
            generate_comparison_plots(results, dirs)
        
        print("\nTraining complete!")
        print(f"Results saved in: {args.data_dir}")
    
    else:
        # Evaluation mode
        print("Starting Evaluation Mode")
        results = evaluate_saved_models(dirs, args.agents)
        generate_comparison_plots(results, dirs)
        print("\nEvaluation complete!")
        print(f"Results saved in: {args.data_dir}")

if __name__ == "__main__":
    main()
