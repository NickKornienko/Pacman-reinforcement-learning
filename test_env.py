"""
Test script to verify enhanced environment functionality
"""
from enhanced_pacman_env import EnhancedPacmanEnv
import time


def test_enhanced_actions():
    env = EnhancedPacmanEnv()
    print("\nTesting Enhanced Pacman Environment")
    print("Testing different action types:")

    # Test basic actions
    print("\n1. Testing basic actions (UP, RIGHT, DOWN, LEFT)")
    state, info = env.reset()
    for action in range(4):
        next_state, reward, done, truncated, info = env.step(action)
        print(f"Basic action {action}: Reward = {reward}, Done = {done}")
        env.render()
        time.sleep(0.5)

    # Test diagonal actions
    print("\n2. Testing diagonal actions")
    state, info = env.reset()
    for action in range(4, 8):
        next_state, reward, done, truncated, info = env.step(action)
        print(f"Diagonal action {action}: Reward = {reward}, Done = {done}")
        env.render()
        time.sleep(0.5)

    # Test speed actions
    print("\n3. Testing speed actions")
    state, info = env.reset()
    for action in range(8, 12):
        next_state, reward, done, truncated, info = env.step(action)
        print(f"Speed action {action}: Reward = {reward}, Done = {done}")
        env.render()
        time.sleep(0.5)

    # Test special actions
    print("\n4. Testing special actions")
    state, info = env.reset()
    for action in range(12, 15):
        next_state, reward, done, truncated, info = env.step(action)
        print(f"Special action {action}: Reward = {reward}, Done = {done}")
        env.render()
        time.sleep(0.5)

    # Test enhanced state information
    print("\n5. Testing enhanced state information")
    state, info = env.reset()
    print("Enhanced state keys:", state.keys())
    print("Ghost distances shape:", len(state['ghost_distances']))
    print("Pellet distances shape:", len(state['pellet_distances']))
    print("State history length:", len(state['history']))

    env.close()
    print("\nEnvironment test complete")


if __name__ == "__main__":
    test_enhanced_actions()
