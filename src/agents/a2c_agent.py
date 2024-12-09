"""
Advantage Actor-Critic (A2C) implementation with move tracking
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from .base_agent import BaseAgent

class A2CNetwork(nn.Module):
    """Neural network for A2C with separate policy and value heads"""
    def __init__(self, state_shape, action_size):
        super(A2CNetwork, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(state_shape[2], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = state_shape[0] * state_shape[1] * 32
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Ensure input is float and normalize
        x = x.float() / 255.0
        
        # Reshape if necessary (NHWC -> NCHW)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared features
        x = self.shared(x)
        
        # Policy and value heads
        policy = F.softmax(self.policy(x), dim=-1)
        value = self.value(x)
        
        return policy, value

class A2CAgent(BaseAgent):
    """A2C Agent with move tracking"""
    def __init__(self, state_shape, action_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(state_shape, action_size)
        
        self.device = device
        print(f"Using device: {device}")
        
        # Network
        self.network = A2CNetwork(state_shape, action_size).to(device)
        
        # Hyperparameters
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        # Training steps
        self.training_steps = 0
        
        # Policy tracking
        self.policy_entropy = []
        self.value_estimates = []
    
    def select_action(self, state):
        """Select action using current policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            policy, value = self.network(state)
            
            # Sample action from policy
            dist = Categorical(policy)
            action = dist.sample()
            
            # Store log probability and value
            log_prob = dist.log_prob(action)
            
            self.states.append(state)
            self.actions.append(action)
            self.values.append(value)
            self.log_probs.append(log_prob)
            
            return action.item()
    
    def update_policy(self, next_state, done):
        """Update policy using collected experience"""
        # Get next state value
        with torch.no_grad():
            _, next_value = self.network(torch.FloatTensor(next_state).to(self.device))
            next_value = next_value if not done else torch.tensor([0.0]).to(self.device)
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = next_value
        
        for r, v in zip(reversed(self.rewards), reversed(self.values)):
            R = r + self.gamma * R
            advantage = R - v
            returns.append(R)
            advantages.append(advantage)
        
        returns = torch.cat(returns[::-1])
        advantages = torch.cat(advantages[::-1])
        
        # Get action probabilities and values
        states = torch.cat(self.states)
        policy, values = self.network(states)
        
        # Calculate action probabilities
        dist = Categorical(policy)
        action_log_probs = dist.log_prob(torch.cat(self.actions))
        dist_entropy = dist.entropy().mean()
        
        # Calculate losses
        policy_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = F.mse_loss(values.squeeze(), returns.detach())
        
        # Total loss
        loss = (policy_loss + 
                self.value_loss_coef * value_loss - 
                self.entropy_coef * dist_entropy)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Clear episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        # Track policy metrics
        self.policy_entropy.append(dist_entropy.item())
        self.value_estimates.append(values.mean().item())
    
    def train(self, env, num_episodes=30):
        """Train the A2C agent"""
        print("\nStarting A2C Training")
        print("Episode Results:")
        print("Episode | Steps | Score | Total Reward | Entropy | Move Distribution")
        print("-" * 75)
        
        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store reward
                self.rewards.append(reward)
                
                # Track move
                self.track_move(action, state, next_state, reward, done or truncated)
                
                episode_reward += reward
                steps += 1
                
                # Update policy if episode ended
                if done or truncated:
                    self.update_policy(next_state, True)
                    break
                
                # Update policy every few steps
                if steps % 5 == 0:
                    self.update_policy(next_state, False)
                
                state = next_state
            
            # End episode tracking
            self.end_episode(info['score'])
            
            # Get move distribution
            move_stats = self.get_move_statistics()
            basic_pct = move_stats['move_distribution']['basic'] * 100
            diag_pct = move_stats['move_distribution']['diagonal'] * 100
            special_pct = move_stats['move_distribution']['special'] * 100
            
            # Print episode results
            print(f"Episode {episode+1:2d} | {steps:3d} | {info['score']:3d} | {episode_reward:6.1f} | "
                  f"{np.mean(self.policy_entropy[-100:]):.3f} | "
                  f"B:{basic_pct:3.0f}% D:{diag_pct:3.0f}% S:{special_pct:3.0f}%")
        
        # Print final summary
        print("\nTraining Complete!")
        print("\nFinal Performance:")
        print(f"Average Score: {np.mean(self.episode_scores):.2f}")
        print(f"Average Steps: {np.mean(self.training_metrics['episode_lengths']):.2f}")
        print(f"Average Reward: {np.mean(self.episode_rewards):.2f}")
        
        # Print move statistics
        final_stats = self.get_move_statistics()
        print("\nMove Usage Statistics:")
        print("Move Type Distribution:")
        for move_type, pct in final_stats['move_distribution'].items():
            print(f"{move_type:8s}: {pct*100:5.1f}%")
        
        print("\nSuccess Rates:")
        for move_type, rate in final_stats['success_rates'].items():
            print(f"{move_type:8s}: {rate*100:5.1f}%")
    
    def save(self, path):
        """Save model and metrics"""
        # Save model
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'policy_entropy': self.policy_entropy,
            'value_estimates': self.value_estimates
        }, path)
        
        # Save metrics
        metrics_path = path.replace('.pt', '_metrics.json')
        self.save_metrics(metrics_path, 'A2C')
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.policy_entropy = checkpoint['policy_entropy']
        self.value_estimates = checkpoint['value_estimates']
