"""
Deep Q-Network (DQN) implementation with move tracking
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import random
from .base_agent import BaseAgent

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Neural network for DQN"""
    def __init__(self, state_shape, action_size):
        super(DQNNetwork, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(state_shape[2], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = state_shape[0] * state_shape[1] * 32
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, action_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Ensure input is float and normalize
        x = x.float() / 255.0
        
        # Reshape if necessary (NHWC -> NCHW)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class DQNAgent(BaseAgent):
    """DQN Agent with move tracking"""
    def __init__(self, state_shape, action_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(state_shape, action_size)
        
        self.device = device
        print(f"Using device: {device}")
        
        # Networks
        self.policy_net = DQNNetwork(state_shape, action_size).to(device)
        self.target_net = DQNNetwork(state_shape, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.target_update = 10
        self.batch_size = 32
        self.update_frequency = 4
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = []
        self.max_memory = 10000
        
        # Training steps
        self.training_steps = 0
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def update_network(self):
        """Update network weights using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        if self.training_steps % self.update_frequency != 0:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in batch])).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def train(self, env, num_episodes=30):
        """Train the DQN agent"""
        print("\nStarting DQN Training")
        print("Episode Results:")
        print("Episode | Steps | Score | Total Reward | Epsilon | Move Distribution")
        print("-" * 75)
        
        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store experience and track move
                self.store_experience(state, action, reward, next_state, done)
                self.track_move(action, state, next_state, reward, done or truncated)
                
                # Update network
                self.update_network()
                self.training_steps += 1
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done or truncated:
                    break
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # End episode tracking
            self.end_episode(info['score'])
            
            # Get move distribution
            move_stats = self.get_move_statistics()
            basic_pct = move_stats['move_distribution']['basic'] * 100
            diag_pct = move_stats['move_distribution']['diagonal'] * 100
            special_pct = move_stats['move_distribution']['special'] * 100
            
            # Print episode results
            print(f"Episode {episode+1:2d} | {steps:3d} | {info['score']:3d} | {episode_reward:6.1f} | "
                  f"{self.epsilon:.3f} | B:{basic_pct:3.0f}% D:{diag_pct:3.0f}% S:{special_pct:3.0f}%")
        
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
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, path)
        
        # Save metrics
        metrics_path = path.replace('.pt', '_metrics.json')
        self.save_metrics(metrics_path, 'DQN')
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
