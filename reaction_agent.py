"""
Reactive agent implementation for Pacman environment.
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

print("\nStarting Reactive Agent on Pacman Environment")
print("Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT")
print("\nEpisode Results:")
print("Episode | Steps | Score | Total Reward")
print("-" * 45)


def get_action(obs):
    pacman_pos = np.argwhere(obs == 2)
    if pacman_pos.size == 0:
        # If Pacman position is not found, take a random action
        return env.action_space.sample()
    pacman_pos = pacman_pos[0]  # Pacman position

    food_pos = np.argwhere(obs == 1)  # Food positions
    ghost_pos = np.argwhere(obs == 3)  # Ghost positions

    if food_pos.size == 0 or ghost_pos.size == 0:
        # If no food or ghosts are found, take a random action
        return env.action_space.sample()

    # Calculate distances to food and ghosts
    food_distances = np.linalg.norm(food_pos - pacman_pos, axis=1)
    ghost_distances = np.linalg.norm(ghost_pos - pacman_pos, axis=1)

    # Find the closest food and ghost
    closest_food = food_pos[np.argmin(food_distances)]
    closest_ghost = ghost_pos[np.argmin(ghost_distances)]

    # Determine action to move towards food and away from ghosts
    action = None
    if np.min(ghost_distances) < 2:  # If a ghost is too close, avoid it
        if closest_ghost[0] < pacman_pos[0]:
            action = 0  # Move UP
        elif closest_ghost[0] > pacman_pos[0]:
            action = 2  # Move DOWN
        elif closest_ghost[1] < pacman_pos[1]:
            action = 1  # Move RIGHT
        elif closest_ghost[1] > pacman_pos[1]:
            action = 3  # Move LEFT
    else:  # Otherwise, move towards the closest food
        if closest_food[0] < pacman_pos[0]:
            action = 0  # Move UP
        elif closest_food[0] > pacman_pos[0]:
            action = 2  # Move DOWN
        elif closest_food[1] < pacman_pos[1]:
            action = 3  # Move LEFT
        elif closest_food[1] > pacman_pos[1]:
            action = 1  # Move RIGHT

    return action


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

            # Select action based on reactive strategy
            action = get_action(obs)

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
        # env.save_animation(f'/pacman_episodes/pacman_episode_{episode+1}.gif')

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
