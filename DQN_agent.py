"""
Advanced DQN agent implementation for Pacman with strategic learning
"""
from pacman_env import PacmanEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from collections import deque

# Create directory for saving episode gifs
os.makedirs("episode_gifs", exist_ok=True)

# Hyperparameters
MAX_NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 500
GAMMA = 0.99
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 5
HIDDEN_SIZE = 512


class StrategicNetwork(nn.Module):
    def __init__(self, action_size):
        super(StrategicNetwork, self).__init__()

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Strategic processing
        self.strategic = nn.Sequential(
            nn.Linear(64 * 20 * 20, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        # Ensure input is float and normalized
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 255.0  # Normalize pixel values

        # Ensure correct input shape and make contiguous
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Extract features
        x = self.features(x)

        # Flatten and ensure contiguous memory layout
        x = x.view(x.size(0), -1)

        # Strategic processing
        x = self.strategic(x)

        # Value and advantage streams
        value = self.value(x)
        advantage = self.advantage(x)

        # Combine streams (Dueling architecture)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class StrategicAgent:
    def __init__(self, state_shape, action_size):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.action_size = action_size
        self.state_shape = state_shape

        # Networks
        self.policy_net = StrategicNetwork(action_size).to(self.device)
        self.target_net = StrategicNetwork(action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=LEARNING_RATE)

        # Memory
        self.memory = ReplayBuffer(MEMORY_SIZE)

        # Training variables
        self.steps_done = 0
        self.episode_rewards = []

    def select_action(self, state, epsilon=0.1):
        try:
            if random.random() > epsilon:
                with torch.no_grad():
                    state = torch.from_numpy(
                        state).float().unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state)
                    return q_values.max(1)[1].item()
            return random.randrange(self.action_size)
        except Exception as e:
            print(f"Error in select_action: {e}")
            print(f"State shape: {state.shape}, dtype: {state.dtype}")
            return random.randrange(self.action_size)

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return

        try:
            # Sample transitions
            transitions = self.memory.sample(BATCH_SIZE)
            batch = list(zip(*transitions))

            # Prepare batch
            state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
            action_batch = torch.LongTensor(batch[1]).to(self.device)
            reward_batch = torch.FloatTensor(batch[2]).to(self.device)
            next_state_batch = torch.FloatTensor(
                np.array(batch[3])).to(self.device)
            done_batch = torch.FloatTensor(batch[4]).to(self.device)

            # Compute current Q values
            current_q_values = self.policy_net(
                state_batch).gather(1, action_batch.unsqueeze(1))

            # Compute next Q values
            with torch.no_grad():
                next_q_values = self.target_net(next_state_batch).max(1)[0]
                target_q_values = reward_batch + \
                    (GAMMA * next_q_values * (1 - done_batch))

            # Compute loss
            loss = F.smooth_l1_loss(
                current_q_values, target_q_values.unsqueeze(1))

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

        except Exception as e:
            print(f"Error in optimize: {e}")

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, env, num_episodes=MAX_NUM_EPISODES):
        print("\nStarting Strategic DQN Training")
        print(f"State shape: {self.state_shape}")
        print(f"Action size: {self.action_size}")
        print("\nEpisode Results:")
        print("Episode | Steps | Score | Total Reward")
        print("-" * 45)

        try:
            for episode in range(num_episodes):
                state, info = env.reset()
                total_reward = 0
                steps = 0

                for step in range(MAX_STEPS_PER_EPISODE):
                    # Select and perform action
                    epsilon = max(0.01, 0.1 - episode/200)
                    action = self.select_action(state, epsilon)
                    next_state, reward, done, truncated, info = env.step(
                        action)

                    # Shape reward based on game state
                    shaped_reward = reward
                    if reward > 0:  # Eating pellets/ghosts
                        shaped_reward *= 2.0
                    elif reward < 0:  # Ghost collision
                        shaped_reward *= 1.5

                    total_reward += shaped_reward
                    steps += 1

                    # Store transition
                    self.memory.push(state, action, shaped_reward,
                                     next_state, done or truncated)
                    state = next_state

                    # Optimize model
                    self.optimize()

                    if done or truncated:
                        break

                # Update target network
                if episode % TARGET_UPDATE_FREQ == 0:
                    self.update_target_network()

                print(
                    f"Episode #{episode+1:2d} | {steps:3d}   | {info['score']:3d}  | {total_reward:5.1f}")
                self.episode_rewards.append(total_reward)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            env.close()

        return self.episode_rewards


def main():
    try:
        env = PacmanEnv()
        state_shape = (20, 20, 3)  # Height, Width, Channels
        action_size = 4  # UP, RIGHT, DOWN, LEFT

        agent = StrategicAgent(state_shape, action_size)
        rewards = agent.train(env)

        print("\nTraining Complete!")
        if rewards:
            print(f"Average Reward: {np.mean(rewards):.2f}")
            print(f"Best Reward: {max(rewards):.2f}")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
