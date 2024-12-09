"""
Visualization script for analyzing and comparing agent performances
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_metrics(metrics_path):
    """Load all metrics files"""
    metrics = {}
    for filename in os.listdir(metrics_path):
        if filename.endswith('_metrics.json'):
            agent_name = filename.split('_metrics')[0]
            with open(os.path.join(metrics_path, filename), 'r') as f:
                data = json.load(f)
                if 'episodes' in data:
                    metrics[agent_name] = {
                        'episode_scores': data['episodes']['score']['values'],
                        'episode_rewards': data['episodes']['reward']['values'],
                        'episode_lengths': data['episodes']['steps']['values']
                    }
                else:
                    metrics[agent_name] = data
    return metrics

def plot_learning_curves(metrics, save_path):
    """Plot learning curves comparing all agents"""
    # Use default style instead of seaborn
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Learning Curves Comparison', fontsize=16)
    
    # Plot scores
    ax = axes[0, 0]
    for agent_name, agent_metrics in metrics.items():
        if agent_name != 'multi_agent':
            ax.plot(agent_metrics['episode_scores'], label=agent_name)
    ax.set_title('Episode Scores')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)
    
    # Plot rewards
    ax = axes[0, 1]
    for agent_name, agent_metrics in metrics.items():
        if agent_name != 'multi_agent':
            ax.plot(agent_metrics['episode_rewards'], label=agent_name)
    ax.set_title('Episode Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True)
    
    # Plot episode lengths
    ax = axes[1, 0]
    for agent_name, agent_metrics in metrics.items():
        if agent_name != 'multi_agent':
            ax.plot(agent_metrics['episode_lengths'], label=agent_name)
    ax.set_title('Episode Lengths')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend()
    ax.grid(True)
    
    # Plot performance distribution
    ax = axes[1, 1]
    data = []
    labels = []
    for agent_name, agent_metrics in metrics.items():
        if agent_name != 'multi_agent':
            data.append(agent_metrics['episode_scores'])
            labels.extend([agent_name] * len(agent_metrics['episode_scores']))
    
    ax.boxplot(data, labels=[name for name in metrics.keys() if name != 'multi_agent'])
    ax.set_title('Score Distribution')
    ax.set_ylabel('Score')
    plt.xticks(rotation=45)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'learning_curves.png'))
    plt.close()

def plot_multi_agent_performance(metrics, save_path):
    """Plot multi-agent system performance"""
    if 'multi_agent' not in metrics:
        return
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Agent System Performance', fontsize=16)
    
    multi_metrics = metrics['multi_agent']
    
    # Plot team scores
    ax = axes[0, 0]
    if isinstance(multi_metrics['episode_scores'][0], list):
        team_scores = [sum(scores) for scores in multi_metrics['episode_scores']]
    else:
        team_scores = multi_metrics['episode_scores']
    ax.plot(team_scores, label='Team Total')
    ax.set_title('Team Score Progression')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)
    
    # Plot individual agent scores
    ax = axes[0, 1]
    if isinstance(multi_metrics['episode_scores'][0], list):
        for i in range(len(multi_metrics['episode_scores'][0])):
            scores = [episode[i] for episode in multi_metrics['episode_scores']]
            ax.plot(scores, label=f'Agent {i+1}')
    ax.set_title('Individual Agent Scores')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)
    
    # Plot episode lengths
    ax = axes[1, 0]
    ax.plot(multi_metrics['episode_lengths'])
    ax.set_title('Episode Lengths')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.grid(True)
    
    # Plot score distribution
    ax = axes[1, 1]
    if isinstance(multi_metrics['episode_scores'][0], list):
        agent_scores = [[scores[i] for scores in multi_metrics['episode_scores']] 
                       for i in range(len(multi_metrics['episode_scores'][0]))]
        ax.boxplot(agent_scores + [team_scores], 
                  labels=[f'Agent {i+1}' for i in range(len(agent_scores))] + ['Team'])
    ax.set_title('Score Distribution')
    ax.set_ylabel('Score')
    plt.xticks(rotation=45)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'multi_agent_performance.png'))
    plt.close()

def plot_move_distributions(metrics, save_path):
    """Plot move type distributions for each agent"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Move Type Distribution', fontsize=16)
    
    move_types = ['basic', 'diagonal', 'special']
    
    for i, (agent_name, agent_metrics) in enumerate(metrics.items()):
        if agent_name != 'multi_agent' and 'episodes' in agent_metrics:
            ax = axes[i//2, i%2]
            
            # Get final move distribution
            move_dist = {
                'basic': agent_metrics['episodes']['basic_moves']['values'][-1],
                'diagonal': agent_metrics['episodes']['diagonal_moves']['values'][-1],
                'special': agent_metrics['episodes']['special_moves']['values'][-1]
            }
            
            ax.bar(move_types, [move_dist[mt] for mt in move_types])
            ax.set_title(f'{agent_name} Move Distribution')
            ax.set_ylabel('Usage %')
            ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'move_distributions.png'))
    plt.close()

def main():
    # Load metrics
    metrics_path = os.path.join('data', 'metrics')
    metrics = load_metrics(metrics_path)
    
    # Create plots directory
    plots_path = os.path.join('data', 'plots')
    os.makedirs(plots_path, exist_ok=True)
    
    # Generate visualizations
    plot_learning_curves(metrics, plots_path)
    plot_multi_agent_performance(metrics, plots_path)
    plot_move_distributions(metrics, plots_path)
    
    print("Visualization complete! Check the plots directory for results.")

if __name__ == "__main__":
    main()
