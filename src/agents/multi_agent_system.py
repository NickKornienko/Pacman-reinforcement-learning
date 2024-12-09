"""
Multi-Agent System implementation with move tracking
Features:
- Multiple agents learning simultaneously
- Shared experience replay
- Inter-agent communication
- Move tracking per agent
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple, defaultdict
import random
from .base_agent import BaseAgent

Experience = namedtuple('Experience', ['agent_id', 'state', 'action', 'reward', 'next_state', 'done'])

class MultiAgentNetwork(nn.Module):
    """Neural network for multi-agent learning with communication"""
    def __init__(self, state_shape, action_size, num_agents=2):
        super(MultiAgentNetwork, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(state_shape[2], 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_shape[0])))
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_shape[1])))
        linear_input_size = convh * convw * 64
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Policy heads (one for each agent)
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            ) for _ in range(num_agents)
        ])
        
        # Value heads (one for each agent)
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(num_agents)
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, states):
        """Process states for all agents"""
        if not isinstance(states, list):
            states = [states]
        
        batch_size = states[0].size(0) if isinstance(states[0], torch.Tensor) else 1
        
        # Process each state
        features = []
        for state in states:
            x = state.float() / 255.0 if isinstance(state, torch.Tensor) else torch.FloatTensor(state).to(next(self.parameters()).device) / 255.0
            
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.permute(0, 3, 1, 2).contiguous()
            
            # Feature extraction
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            
            # Shared features
            x = self.shared(x)
            features.append(x)
        
        # Generate policies and values for each agent
        policies = []
        values = []
        
        for i, feature in enumerate(features):
            # Policy
            policy = F.softmax(self.policy_heads[i](feature), dim=-1)
            policies.append(policy)
            
            # Value
            value = self.value_heads[i](feature)
            values.append(value)
        
        return policies, values

class MultiAgentSystem(BaseAgent):
    """Multi-Agent System with move tracking"""
    def __init__(self, state_shape, action_size, num_agents=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(state_shape, action_size)
        
        self.num_agents = num_agents
        self.device = device
        print(f"Using device: {device}")
        
        # Networks
        self.policy_net = MultiAgentNetwork(state_shape, action_size, num_agents).to(device)
        self.target_net = MultiAgentNetwork(state_shape, action_size, num_agents).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Hyperparameters
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.update_frequency = 4
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        
        # Move tracking per agent
        self.agent_move_trackers = [defaultdict(list) for _ in range(num_agents)]
        self.agent_success_rates = [defaultdict(float) for _ in range(num_agents)]
        self.agent_interactions = defaultdict(int)  # Track when agents use similar moves
        
        # Training metrics
        self.training_steps = 0
        self.episode_rewards = [[] for _ in range(num_agents)]
        self.episode_scores = [[] for _ in range(num_agents)]
        self.episode_moves = [[] for _ in range(num_agents)]
        self.training_metrics = {'episode_lengths': []}
    
    def select_actions(self, states):
        """Select actions for all agents"""
        with torch.no_grad():
            # Convert states to tensors
            state_tensors = [
                torch.FloatTensor(state).to(self.device) if isinstance(state, np.ndarray)
                else state.to(self.device)
                for state in states
            ]
            
            # Get policies
            policies, _ = self.policy_net(state_tensors)
            
            # Sample actions
            actions = []
            for policy in policies:
                dist = torch.distributions.Categorical(policy)
                action = dist.sample()
                actions.append(action.item())
            
            return actions
    
    def store_experience(self, experiences):
        """Store experiences and track moves"""
        for exp in experiences:
            # Store experience
            self.memory.append(Experience(*exp))
            
            # Track move
            agent_id = exp[0]
            action = exp[2]
            reward = exp[3]
            
            # Track move type
            if action < 4:
                move_type = 'basic'
            elif action < 8:
                move_type = 'diagonal'
            else:
                move_type = 'special'
            
            self.agent_move_trackers[agent_id][move_type].append(action)
            
            # Track success
            if reward > 0:
                self.agent_success_rates[agent_id][move_type] += 1
            
            # Track agent interactions (when agents use similar move types)
            for other_id in range(self.num_agents):
                if other_id != agent_id:
                    other_action = experiences[other_id][2]
                    if (other_action < 4 and action < 4) or \
                       (4 <= other_action < 8 and 4 <= action < 8) or \
                       (other_action >= 8 and action >= 8):
                        self.agent_interactions[(agent_id, other_id)] += 1
    
    def update_networks(self):
        """Update networks using shared experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        if self.training_steps % self.update_frequency != 0:
            self.training_steps += 1
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Organize batch by agent
        agent_batches = [[] for _ in range(self.num_agents)]
        for exp in batch:
            agent_batches[exp.agent_id].append(exp)
        
        # Process each agent's batch
        total_loss = 0
        for agent_id, agent_batch in enumerate(agent_batches):
            if not agent_batch:
                continue
            
            # Convert batch to numpy array first
            states = np.array([exp.state for exp in agent_batch])
            next_states = np.array([exp.next_state for exp in agent_batch])
            
            # Prepare tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor([exp.action for exp in agent_batch]).to(self.device)
            rewards = torch.FloatTensor([exp.reward for exp in agent_batch]).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor([exp.done for exp in agent_batch]).to(self.device)
            
            # Get current Q values
            policies, values = self.policy_net([states])
            current_values = values[0].squeeze()  # Use first value since we process one agent at a time
            
            # Get next values
            with torch.no_grad():
                _, next_values = self.target_net([next_states])
                target_values = rewards + (1 - dones) * self.gamma * next_values[0].squeeze()
            
            # Compute value loss
            value_loss = F.mse_loss(current_values, target_values)
            
            # Compute policy loss
            log_probs = torch.log(policies[0].gather(1, actions.unsqueeze(1))).squeeze()
            advantages = target_values.detach() - current_values.detach()
            policy_loss = -(log_probs * advantages).mean()
            
            # Total loss
            loss = value_loss + policy_loss
            total_loss += loss
        
        # Update networks
        if total_loss > 0:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
        
        self.training_steps += 1
        
        # Update target network occasionally
        if self.training_steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def get_agent_statistics(self, agent_id):
        """Get move statistics for a specific agent"""
        move_tracker = self.agent_move_trackers[agent_id]
        success_rates = self.agent_success_rates[agent_id]
        
        total_moves = sum(len(moves) for moves in move_tracker.values())
        if total_moves == 0:
            return {
                'move_distribution': {
                    'basic': 0.0,
                    'diagonal': 0.0,
                    'special': 0.0
                },
                'success_rates': {
                    'basic': 0.0,
                    'diagonal': 0.0,
                    'special': 0.0
                }
            }
        
        stats = {
            'move_distribution': {
                move_type: len(moves) / total_moves
                for move_type, moves in move_tracker.items()
            },
            'success_rates': {
                move_type: success_rates[move_type] / len(moves) if len(moves) > 0 else 0
                for move_type, moves in move_tracker.items()
            }
        }
        return stats
    
    def get_interaction_statistics(self):
        """Get statistics about agent interactions"""
        total_interactions = sum(self.agent_interactions.values())
        if total_interactions == 0:
            return {}
        
        stats = {
            'interaction_rates': {
                f'agent_{i}_agent_{j}': count / total_interactions
                for (i, j), count in self.agent_interactions.items()
            }
        }
        return stats
    
    def save(self, path):
        """Save model and metrics"""
        # Save model
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
            'episode_scores': self.episode_scores,
            'agent_move_trackers': self.agent_move_trackers,
            'agent_success_rates': self.agent_success_rates,
            'agent_interactions': dict(self.agent_interactions)
        }, path)
    
    def load(self, path):
        """Load model and metrics"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_scores = checkpoint['episode_scores']
        self.agent_move_trackers = checkpoint['agent_move_trackers']
        self.agent_success_rates = checkpoint['agent_success_rates']
        self.agent_interactions = defaultdict(int, checkpoint['agent_interactions'])
