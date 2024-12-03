"""
Evaluation and comparison module for DQN variants and A2C
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import time
import os
import pygame
import gc
from DQN_agent import StrategicAgent
from sophisticated_DQN_agent import SophisticatedAgent
from dueling_DQN_agent import DuelingDQNAgent
from a2c_agent import A2CAgent
from pacman_env import PacmanEnv
from pacman_wrapper import SophisticatedPacmanEnv

# Create directories for saving
os.makedirs("trained_models", exist_ok=True)
os.makedirs("episode_gifs", exist_ok=True)
os.makedirs("evaluation_results", exist_ok=True)

class AgentEvaluator:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.episode_data = defaultdict(list)
    
    def save_model(self, agent, agent_name):
        """Save trained model"""
        model_path = f"trained_models/{agent_name.lower().replace(' ', '_')}.pth"
        if isinstance(agent, A2CAgent):
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
            }, model_path)
        else:
            torch.save({
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, model_path)
        print(f"Saved model to {model_path}")
    
    def load_model(self, agent, agent_name):
        """Load trained model"""
        model_path = f"trained_models/{agent_name.lower().replace(' ', '_')}.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            if isinstance(agent, A2CAgent):
                agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            else:
                agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded pre-trained model from {model_path}")
            return True
        return False
    
    def train_agent(self, agent, env_class, train_episodes=30, agent_name=""):
        """Train an agent"""
        print(f"\nTraining {agent_name} for {train_episodes} episodes...")
        
        try:
            # Create environment
            env = env_class()
            agent.train(env, num_episodes=train_episodes)
            self.save_model(agent, agent_name)
            gc.collect()
            return True
        except Exception as e:
            print(f"Error during {agent_name} training: {e}")
            return False
        finally:
            env.close()
    
    def evaluate_agent(self, agent, env_class, eval_episodes=10, render=True, agent_name=""):
        """Evaluate a pre-trained agent"""
        try:
            print(f"\nEvaluating {agent_name}...")
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
            
            # Create environment
            env = env_class()
            
            for episode in range(eval_episodes):
                state, info = env.reset()
                total_reward = 0
                steps = 0
                
                while True:
                    if render:
                        env.render()
                        time.sleep(0.02)
                    
                    action = agent.select_action(state, epsilon=0.01)
                    next_state, reward, done, truncated, info = env.step(action)
                    
                    total_reward += reward
                    steps += 1
                    state = next_state
                    
                    if done or truncated or steps >= 1000:
                        if info['score'] > 500:
                            wins += 1
                        if steps < 500:
                            completions += 1
                        break
                
                evaluation_metrics['rewards'].append(total_reward)
                evaluation_metrics['steps'].append(steps)
                evaluation_metrics['scores'].append(info['score'])
                
                if render and episode % 2 == 0:
                    try:
                        gif_path = f'episode_gifs/{agent_name.lower().replace(" ", "_")}_eval_{episode+1}.gif'
                        env.save_animation(gif_path)
                        print(f"Saved animation to {gif_path}")
                    except:
                        pass
                
                print(f"Episode {episode+1}/{eval_episodes} - Score: {info['score']} - Steps: {steps}")
                
                if episode % 3 == 0:
                    gc.collect()
            
            evaluation_metrics['win_rate'] = wins / eval_episodes
            evaluation_metrics['avg_score'] = np.mean(evaluation_metrics['scores'])
            evaluation_metrics['completion_rate'] = completions / eval_episodes
            
            return evaluation_metrics
            
        except Exception as e:
            print(f"Error during {agent_name} evaluation: {e}")
            return {
                'rewards': [],
                'steps': [],
                'scores': [],
                'win_rate': 0,
                'avg_score': 0,
                'completion_rate': 0
            }
        finally:
            env.close()
            gc.collect()

    def train_and_evaluate_all(self, train_episodes=30, eval_episodes=10):
        """Train and evaluate all agents"""
        try:
            # Initialize agents
            state_shape = (20, 20, 3)
            basic_agent = StrategicAgent(state_shape, action_size=4)
            sophisticated_agent = SophisticatedAgent(state_shape)
            dueling_agent = DuelingDQNAgent(state_shape)
            a2c_agent = A2CAgent(state_shape)
            
            # Train all agents
            print("\nStarting training phase...")
            if not all([
                self.train_agent(basic_agent, PacmanEnv, train_episodes, "Basic DQN"),
                self.train_agent(sophisticated_agent, SophisticatedPacmanEnv, train_episodes, "Sophisticated DQN"),
                self.train_agent(dueling_agent, SophisticatedPacmanEnv, train_episodes, "Dueling DQN"),
                self.train_agent(a2c_agent, SophisticatedPacmanEnv, train_episodes, "A2C")
            ]):
                print("Error: Training failed for one or more agents.")
                return
            
            # Load trained models for evaluation
            print("\nLoading trained models for evaluation...")
            if not all([
                self.load_model(basic_agent, "Basic DQN"),
                self.load_model(sophisticated_agent, "Sophisticated DQN"),
                self.load_model(dueling_agent, "Dueling DQN"),
                self.load_model(a2c_agent, "A2C")
            ]):
                print("Error: Could not load all trained models.")
                return
            
            # Evaluate each agent
            print("\nStarting evaluation phase...")
            basic_metrics = self.evaluate_agent(
                basic_agent, PacmanEnv, eval_episodes, agent_name="Basic DQN")
            gc.collect()
            
            sophisticated_metrics = self.evaluate_agent(
                sophisticated_agent, SophisticatedPacmanEnv, eval_episodes, agent_name="Sophisticated DQN")
            gc.collect()
            
            dueling_metrics = self.evaluate_agent(
                dueling_agent, SophisticatedPacmanEnv, eval_episodes, agent_name="Dueling DQN")
            gc.collect()
            
            a2c_metrics = self.evaluate_agent(
                a2c_agent, SophisticatedPacmanEnv, eval_episodes, agent_name="A2C")
            gc.collect()
            
            # Store results
            self.metrics['basic_dqn'] = basic_metrics
            self.metrics['sophisticated_dqn'] = sophisticated_metrics
            self.metrics['dueling_dqn'] = dueling_metrics
            self.metrics['a2c'] = a2c_metrics
            
            # Generate comparison visualizations
            self.plot_comparison()
            
            # Save metrics to file
            self.save_metrics()
            
            # Print comparison summary
            print("\nPerformance Comparison:")
            print("-" * 80)
            print(f"Metric              | Basic DQN  | Sophisticated DQN | Dueling DQN | A2C")
            print("-" * 80)
            print(f"Average Score       | {basic_metrics['avg_score']:.2f}      | {sophisticated_metrics['avg_score']:.2f}            | {dueling_metrics['avg_score']:.2f}        | {a2c_metrics['avg_score']:.2f}")
            print(f"Win Rate           | {basic_metrics['win_rate']:.2%}      | {sophisticated_metrics['win_rate']:.2%}            | {dueling_metrics['win_rate']:.2%}        | {a2c_metrics['win_rate']:.2%}")
            print(f"Completion Rate    | {basic_metrics['completion_rate']:.2%}      | {sophisticated_metrics['completion_rate']:.2%}            | {dueling_metrics['completion_rate']:.2%}        | {a2c_metrics['completion_rate']:.2%}")
            print(f"Avg Steps/Episode  | {np.mean(basic_metrics['steps']):.2f}      | {np.mean(sophisticated_metrics['steps']):.2f}            | {np.mean(dueling_metrics['steps']):.2f}        | {np.mean(a2c_metrics['steps']):.2f}")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error during training and evaluation: {e}")
        finally:
            gc.collect()

    def save_metrics(self):
        """Save evaluation metrics to file"""
        results_path = "evaluation_results/metrics.txt"
        with open(results_path, 'w') as f:
            f.write("Performance Comparison:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Metric              | Basic DQN  | Sophisticated DQN | Dueling DQN | A2C\n")
            f.write("-" * 80 + "\n")
            
            for agent_type in ['basic_dqn', 'sophisticated_dqn', 'dueling_dqn', 'a2c']:
                metrics = self.metrics[agent_type]
                f.write(f"Agent: {agent_type}\n")
                f.write(f"Average Score: {metrics['avg_score']:.2f}\n")
                f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
                f.write(f"Completion Rate: {metrics['completion_rate']:.2%}\n")
                f.write(f"Average Steps: {np.mean(metrics['steps']):.2f}\n\n")

    def plot_comparison(self):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        for agent_type in ['basic_dqn', 'sophisticated_dqn', 'dueling_dqn', 'a2c']:
            label = agent_type.replace('_', ' ').title()
            axes[0, 0].plot(self.metrics[agent_type]['rewards'], label=label)
        axes[0, 0].set_title('Rewards per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        
        # Plot scores
        for agent_type in ['basic_dqn', 'sophisticated_dqn', 'dueling_dqn', 'a2c']:
            label = agent_type.replace('_', ' ').title()
            axes[0, 1].plot(self.metrics[agent_type]['scores'], label=label)
        axes[0, 1].set_title('Scores per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        
        # Plot steps
        for agent_type in ['basic_dqn', 'sophisticated_dqn', 'dueling_dqn', 'a2c']:
            label = agent_type.replace('_', ' ').title()
            axes[1, 0].plot(self.metrics[agent_type]['steps'], label=label)
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].legend()
        
        # Bar plot of win rates and completion rates
        metrics = ['win_rate', 'completion_rate']
        x = np.arange(len(metrics))
        width = 0.2
        
        axes[1, 1].bar(x - 1.5*width, [self.metrics['basic_dqn'][m] for m in metrics], width, label='Basic DQN')
        axes[1, 1].bar(x - 0.5*width, [self.metrics['sophisticated_dqn'][m] for m in metrics], width, label='Sophisticated DQN')
        axes[1, 1].bar(x + 0.5*width, [self.metrics['dueling_dqn'][m] for m in metrics], width, label='Dueling DQN')
        axes[1, 1].bar(x + 1.5*width, [self.metrics['a2c'][m] for m in metrics], width, label='A2C')
        
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Win Rate', 'Completion Rate'])
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('evaluation_results/comparison_plots.png', dpi=100)
        plt.close()
        gc.collect()

def main():
    try:
        evaluator = AgentEvaluator()
        evaluator.train_and_evaluate_all(train_episodes=30, eval_episodes=10)
        print("\nTraining and evaluation complete! Results saved to evaluation_results/")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        gc.collect()

if __name__ == "__main__":
    main()
