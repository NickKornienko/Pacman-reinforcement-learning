"""
Evaluation and comparison module for DQN variants
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import time
from DQN_agent import StrategicAgent
from sophisticated_DQN_agent import SophisticatedAgent
from pacman_env import PacmanEnv
from pacman_wrapper import SophisticatedPacmanEnv

class AgentEvaluator:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.episode_data = defaultdict(list)
        
    def evaluate_agent(self, agent, env, num_episodes=50, render=True):
        """Evaluate an agent's performance"""
        evaluation_metrics = {
            'rewards': [],
            'steps': [],
            'scores': [],
            'win_rate': 0,
            'avg_score': 0,
            'completion_rate': 0
        }
        
        wins = 0
        completions = 0
        
        for episode in range(num_episodes):
            state, info = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                if render:
                    env.render()
                    time.sleep(0.05)
                
                # Use epsilon=0.01 for evaluation (minimal exploration)
                action = agent.select_action(state, epsilon=0.01)
                next_state, reward, done, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done or truncated:
                    if info['score'] > 500:  # Consider this a win
                        wins += 1
                    if steps < 500:  # Consider this a completion
                        completions += 1
                    break
            
            evaluation_metrics['rewards'].append(total_reward)
            evaluation_metrics['steps'].append(steps)
            evaluation_metrics['scores'].append(info['score'])
            
            if render:
                env.save_animation(f'episode_gifs/eval_episode_{episode+1}.gif')
        
        # Calculate final metrics
        evaluation_metrics['win_rate'] = wins / num_episodes
        evaluation_metrics['avg_score'] = np.mean(evaluation_metrics['scores'])
        evaluation_metrics['completion_rate'] = completions / num_episodes
        
        return evaluation_metrics

    def compare_agents(self, num_episodes=50):
        """Compare DQN and Sophisticated DQN agents"""
        # Initialize environments and agents
        basic_env = PacmanEnv()
        sophisticated_env = SophisticatedPacmanEnv()
        
        state_shape = (20, 20, 3)
        basic_agent = StrategicAgent(state_shape, action_size=4)
        sophisticated_agent = SophisticatedAgent(state_shape)
        
        print("\nEvaluating Basic DQN Agent...")
        basic_metrics = self.evaluate_agent(basic_agent, basic_env, num_episodes)
        
        print("\nEvaluating Sophisticated DQN Agent...")
        sophisticated_metrics = self.evaluate_agent(sophisticated_agent, sophisticated_env, num_episodes)
        
        # Store results
        self.metrics['basic_dqn'] = basic_metrics
        self.metrics['sophisticated_dqn'] = sophisticated_metrics
        
        # Generate comparison visualizations
        self.plot_comparison()
        
        # Print comparison summary
        print("\nPerformance Comparison:")
        print("-" * 50)
        print(f"Metric              | Basic DQN  | Sophisticated DQN")
        print("-" * 50)
        print(f"Average Score       | {basic_metrics['avg_score']:.2f}      | {sophisticated_metrics['avg_score']:.2f}")
        print(f"Win Rate           | {basic_metrics['win_rate']:.2%}      | {sophisticated_metrics['win_rate']:.2%}")
        print(f"Completion Rate    | {basic_metrics['completion_rate']:.2%}      | {sophisticated_metrics['completion_rate']:.2%}")
        print(f"Avg Steps/Episode  | {np.mean(basic_metrics['steps']):.2f}      | {np.mean(sophisticated_metrics['steps']):.2f}")
        print("-" * 50)

    def plot_comparison(self):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axes[0, 0].plot(self.metrics['basic_dqn']['rewards'], label='Basic DQN')
        axes[0, 0].plot(self.metrics['sophisticated_dqn']['rewards'], label='Sophisticated DQN')
        axes[0, 0].set_title('Rewards per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        
        # Plot scores
        axes[0, 1].plot(self.metrics['basic_dqn']['scores'], label='Basic DQN')
        axes[0, 1].plot(self.metrics['sophisticated_dqn']['scores'], label='Sophisticated DQN')
        axes[0, 1].set_title('Scores per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        
        # Plot steps
        axes[1, 0].plot(self.metrics['basic_dqn']['steps'], label='Basic DQN')
        axes[1, 0].plot(self.metrics['sophisticated_dqn']['steps'], label='Sophisticated DQN')
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].legend()
        
        # Bar plot of win rates and completion rates
        metrics = ['win_rate', 'completion_rate']
        basic_rates = [self.metrics['basic_dqn'][m] for m in metrics]
        soph_rates = [self.metrics['sophisticated_dqn'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, basic_rates, width, label='Basic DQN')
        axes[1, 1].bar(x + width/2, soph_rates, width, label='Sophisticated DQN')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Win Rate', 'Completion Rate'])
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png')
        plt.close()

def main():
    try:
        evaluator = AgentEvaluator()
        evaluator.compare_agents(num_episodes=50)
        print("\nEvaluation complete! Results saved to evaluation_results.png")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
