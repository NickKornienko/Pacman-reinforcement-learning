"""
DQN agent implementation for Pacman environment.
"""
from pacman_env import PacmanEnv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Create Pacman environment
env = PacmanEnv()

# Hyperparameters
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

# Neural network for Q-function


class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        # Convolutional layers to process the 3D input
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers after flattening
        self.fc1 = nn.Linear(64 * 20 * 20, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        # Input shape should be (batch_size, channels, height, width)
        # Rearrange to (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Initialize DQN
action_size = env.action_space.n
policy_net = DQN(action_size)
target_net = DQN(action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

# Epsilon-greedy policy
epsilon = EPSILON_START


def select_action(state):
    global epsilon
    # Convert state to tensor and add batch dimension
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return policy_net(state).argmax().item()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    current_q_values = policy_net(states).gather(1, actions).squeeze()
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    loss = nn.functional.mse_loss(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Store results for documentation
results = {
    'episode_steps': [],
    'episode_rewards': [],
    'episode_scores': []
}

print("\nStarting DQN Agent on Pacman Environment")
print("Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT")
print("\nEpisode Results:")
print("Episode | Steps | Score | Total Reward")
print("-" * 45)

try:
    # Run episodes
    for episode in range(MAX_NUM_EPISODES):
        obs, info = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            # Render environment
            env.render()

            # Small delay to make visualization visible
            time.sleep(0.1)

            # Select action
            action = select_action(obs)

            # Execute action
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            # Store transition in memory
            memory.append((obs, action, reward, next_state, done or truncated))

            # Update observation
            obs = next_state

            # Optimize the model
            optimize_model()

            if done or truncated:
                print(
                    f"Episode #{episode+1:2d} | {step+1:3d}   | {info['score']:3d}  | {episode_reward:5.1f}")
                results['episode_steps'].append(step+1)
                results['episode_rewards'].append(episode_reward)
                results['episode_scores'].append(info['score'])
                break

        # Save animation after each episode
        env.save_animation(f'pacman_episode_{episode+1}.gif')

        # Update epsilon
        epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon)

        # Update target network
        if episode % 2 == 0:
            target_net.load_state_dict(policy_net.state_dict())

finally:
    env.close()

# Print summary statistics
print("\nResults Summary:")
print(f"Average Steps per Episode: {np.mean(results['episode_steps']):.2f}")
print(f"Average Score per Episode: {np.mean(results['episode_scores']):.2f}")
print(f"Average Reward per Episode: {np.mean(results['episode_rewards']):.2f}")
print(f"Best Score: {max(results['episode_scores'])}")
print(f"Best Reward: {max(results['episode_rewards']):.2f}")
print("\nAnimation files have been saved as 'pacman_episode_X.gif' for each episode")
