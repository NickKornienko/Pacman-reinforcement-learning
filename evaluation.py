"""
Evaluation and comparison module for DQN variants
"""
# [Previous imports remain the same...]

def main():
    try:
        evaluator = AgentEvaluator()
        # Reduced training episodes for faster completion
        evaluator.compare_agents(train_episodes=30, eval_episodes=10)
        print("\nEvaluation complete! Results saved to evaluation_results/")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
