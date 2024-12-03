"""
Actor-Critic (A2C) implementation for Pacman using policy gradients
"""
from enhanced_pacman_env import EnhancedPacmanEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os
from collections import deque
import time

# Create directories for saving
os.makedirs("trained_models", exist_ok=True)
os.makedirs("episode_gifs", exist_ok=True)

class SharedFeatures(nn.Module):
    def __init__(self, input_shape):
        super(SharedFeatures, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.conv_out_size = 64 * input_shape[0] * input_shape[1]
        
        self.strategic = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
    
    def forward(self, x):
        if isinstance(x, dict):
            x = x['pixels']
        
        x = torch.as_tensor(x, dtype=torch.float32)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).contiguous() / 255.0
        
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.strategic(x)

class Actor(nn.Module):
    def __init__(self, input_shape, action_size):
        super(Actor, self).__init__()
        
        self.shared = SharedFeatures(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        return self.policy(shared_features)

class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        
        self.shared = SharedFeatures(input_shape)
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        return self.value(shared_features)

class A2CAgent:
    def __init__(self, state_shape):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.state_shape = state_shape
        self.action_size = 15
        
        # Networks
        self.actor = Actor(state_shape[:2], self.action_size).to(self.device)
        self.critic = Critic(state_shape[:2]).to(self.device)
        self.policy_net = self.actor  # For compatibility with evaluator
        self.target_net = self.actor  # For compatibility with evaluator
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.optimizer = self.actor_optimizer  # For compatibility with evaluator
        
        # Training components
        self.gamma = 0.99
        self.entropy_beta = 0.01
        self.episode_actions = []
        self.successful_sequences = deque(maxlen=100)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
    
    def select_action(self, state, epsilon=0.0):
        with torch.no_grad():
            if isinstance(state, dict):
                state_tensor = torch.FloatTensor(state['pixels']).to(self.device)
            else:
                state_tensor = torch.FloatTensor(state).to(self.device)
            
            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(0)
            
            action_probs = self.actor(state_tensor)
            
            if np.random.random() < epsilon:
                action = np.random.randint(self.action_size)
                dist = Categorical(action_probs)
                log_prob = dist.log_prob(torch.tensor([action]).to(self.device))
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = action.item()
            
            self.episode_actions.append(action)
            self.log_probs.append(log_prob)
            return action
    
    def optimize(self):
        if len(self.states) == 0:
            return
        
        # Convert states to tensors
        states = [s['pixels'] if isinstance(s, dict) else s for s in self.states]
        next_states = [s['pixels'] if isinstance(s, dict) else s for s in self.next_states]
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        # Get current state values
        values = self.critic(states)
        
        # Get next state values for bootstrapping
        with torch.no_grad():
            next_values = self.critic(next_states)
            # Calculate returns using one-step TD
            returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)
            returns = returns.detach()
        
        # Calculate advantages
        advantages = returns - values.squeeze()
        
        # Actor loss
        policy_loss = 0
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss -= log_prob * advantage.detach()  # Detach advantages
        
        # Add entropy bonus
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        entropy = dist.entropy().mean()
        
        # Critic loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Total loss for actor
        actor_loss = policy_loss.mean() - self.entropy_beta * entropy
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
    
    def save_model(self):
        """Save the trained model"""
        model_path = "trained_models/a2c_agent.pth"
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, model_path)
        print(f"Saved model to {model_path}")
    
    def train(self, env, num_episodes=20):  # Changed to 20 episodes to match evaluation
        print("\nStarting A2C Training")
        print("Using enhanced action space with diagonal and special movements")
        print("\nEpisode Results:")
        print("Episode | Steps | Score | Total Reward | Action Types")
        print("-" * 65)
        
        try:
            for episode in range(num_episodes):
                state, info = env.reset()
                total_reward = 0
                steps = 0
                self.episode_actions = []
                
                env.render()
                
                for step in range(500):
                    env.render()
                    time.sleep(0.1)
                    
                    epsilon = max(0.01, 0.1 - episode/10)  # Adjusted epsilon decay
                    action = self.select_action(state, epsilon)
                    next_state, reward, done, truncated, info = env.step(int(action))
                    
                    shaped_reward = reward
                    if action >= 4:  # Complex action bonus
                        shaped_reward *= 1.2
                    if reward > 0:  # Eating pellets/ghosts
                        shaped_reward *= 2.0
                    elif reward < 0:  # Ghost collision
                        shaped_reward *= 1.5
                    
                    total_reward += shaped_reward
                    steps += 1
                    
                    self.states.append(state)
                    self.actions.append(action)
                    self.rewards.append(shaped_reward)
                    self.next_states.append(next_state)
                    self.dones.append(done or truncated)
                    
                    state = next_state
                    
                    if len(self.states) >= 32 or done or truncated:
                        self.optimize()
                    
                    if done or truncated:
                        break
                
                env.save_animation(f'episode_gifs/a2c_episode_{episode+1}.gif')
                
                action_counts = np.bincount(self.episode_actions, minlength=self.action_size)
                basic_actions = sum(action_counts[:4])
                complex_actions = sum(action_counts[4:])
                
                print(f"Episode #{episode+1:2d} | {steps:3d} | {info['score']:3d} | {total_reward:5.1f} | "
                      f"Basic: {basic_actions:2d}, Complex: {complex_actions:2d}")
                
                if info['score'] > 100:
                    self.successful_sequences.append(list(self.episode_actions))
            
            # Save the trained model
            self.save_model()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_model()
        except Exception as e:
            print(f"Error during training: {e}")
            raise e
        finally:
            env.close()

def main():
    try:
        env = EnhancedPacmanEnv()
        state_shape = (20, 20, 3)
        agent = A2CAgent(state_shape)
        agent.train(env)
    except Exception as e:
        print(f"Error in main: {e}")
        raise e

if __name__ == "__main__":
    main()
