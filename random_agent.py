"""
Random agent implementation for Pacman environment.
"""
from pacman_env import PacmanEnv
import time
import numpy as np

# Create Pacman environment
env = PacmanEnv()


MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500

# Store results for documentation
results = {
    'episode_steps': [],
    'episode_rewards': [],
    'episode_scores': []
}

print("\nStarting Random Agent on Pacman Environment")
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

            # Sample random action
            action = env.action_space.sample()

            # Execute action
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            # Update observation
            obs = next_state

            if done or truncated:
                print(
                    f"Episode #{episode+1:2d} | {step+1:3d}   | {info['score']:3d}  | {episode_reward:5.1f}")
                results['episode_steps'].append(step+1)
                results['episode_rewards'].append(episode_reward)
                results['episode_scores'].append(info['score'])
                break

        # Save animation after each episode
        env.save_animation(f'pacman_episode_{episode+1}.gif')

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
