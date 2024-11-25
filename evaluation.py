def main():
    try:
        evaluator = AgentEvaluator()
        # Set to 20 training episodes for faster completion
        evaluator.compare_agents(train_episodes=20, eval_episodes=10)
        print("\nEvaluation complete! Results saved to evaluation_results/")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        try:
            pygame.quit()
        except:
            pass

if __name__ == "__main__":
    main()
