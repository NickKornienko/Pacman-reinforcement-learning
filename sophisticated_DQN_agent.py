"""
Sophisticated DQN agent using enhanced Pacman environment
"""
from pacman_wrapper import SophisticatedPacmanEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from collections import deque
import time

# Create directory for saving episode gifs
os.makedirs("episode_gifs", exist_ok=True)


class SophisticatedNetwork(nn.Module):
    def __init__(self, input_shape, action_size):
        super(SophisticatedNetwork, self).__init__()

        # Feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Calculate size after convolutions
        conv_out_size = 64 * input_shape[0] * input_shape[1]

        # Strategic layers
        self.strategic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Value and Advantage streams
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, action_size)

    def forward(self, x):
        # Ensure correct input shape and normalization
        x = x.permute(0, 3, 1, 2).contiguous().float() / 255.0

        # Process through layers
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view
        x = self.strategic(x)

        # Value and Advantage streams
        value = self.value(x)
        advantage = self.advantage(x)

        # Combine streams
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


class SophisticatedAgent:
    def __init__(self, state_shape):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.state_shape = state_shape
        self.action_size = 15  # Total number of sophisticated actions

        # Networks
        self.policy_net = SophisticatedNetwork(
            state_shape[:2], self.action_size).to(self.device)
        self.target_net = SophisticatedNetwork(
            state_shape[:2], self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Training components
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = 0.99

        # Episode history for analysis
        self.episode_actions = []
        self.successful_sequences = deque(maxlen=100)

    def select_action(self, state, epsilon=0.1):
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(
                    state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
                self.episode_actions.append(action)
                return action
        action = random.randrange(self.action_size)
        self.episode_actions.append(action)
        return action

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        # Convert to tensors and ensure contiguous memory layout
        state_batch = torch.FloatTensor(
            np.array(batch[0])).contiguous().to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(
            np.array(batch[3])).contiguous().to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)

        # Compute current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch)

        # Compute next Q values
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + (self.gamma * next_q * (1 - done_batch))

        # Compute loss and optimize
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def train(self, env, num_episodes=1000):
        print("\nStarting Sophisticated DQN Training")
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

                # Initial render
                env.render()

                for step in range(500):
                    # Render current state
                    env.render()
                    time.sleep(0.1)

                    # Select and perform action
                    epsilon = max(0.01, 0.1 - episode/200)
                    action = self.select_action(state, epsilon)
                    next_state, reward, done, truncated, info = env.step(
                        action)

                    # Shape reward based on action type
                    shaped_reward = reward
                    if action >= 4:  # Complex action bonus
                        shaped_reward *= 1.2
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

                # Save episode animation
                env.save_animation(f'episode_gifs/episode_{episode+1}.gif')

                # Analyze action distribution
                action_counts = np.bincount(
                    self.episode_actions, minlength=self.action_size)
                basic_actions = sum(action_counts[:4])
                complex_actions = sum(action_counts[4:])

                # Update target network
                if episode % 10 == 0:
                    self.target_net.load_state_dict(
                        self.policy_net.state_dict())

                # Print episode results with action distribution
                print(f"Episode #{episode+1:2d} | {steps:3d} | {info['score']:3d} | {total_reward:5.1f} | "
                      f"Basic: {basic_actions:2d}, Complex: {complex_actions:2d}")

                # Save successful action sequences
                if info['score'] > 100:
                    self.successful_sequences.append(
                        list(self.episode_actions))

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            env.close()


def main():
    try:
        env = SophisticatedPacmanEnv()
        state_shape = (20, 20, 3)  # Height, Width, Channels
        agent = SophisticatedAgent(state_shape)
        agent.train(env)
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
