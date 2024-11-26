def plot_comparison(self):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        for agent_type in ['basic_dqn', 'sophisticated_dqn', 'dueling_dqn']:
            label = agent_type.replace('_', ' ').title()
            axes[0, 0].plot(self.metrics[agent_type]['rewards'], label=label)
        axes[0, 0].set_title('Rewards per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        
        # Plot scores
        for agent_type in ['basic_dqn', 'sophisticated_dqn', 'dueling_dqn']:
            label = agent_type.replace('_', ' ').title()
            axes[0, 1].plot(self.metrics[agent_type]['scores'], label=label)
        axes[0, 1].set_title('Scores per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        
        # Plot steps
        for agent_type in ['basic_dqn', 'sophisticated_dqn', 'dueling_dqn']:
            label = agent_type.replace('_', ' ').title()
            axes[1, 0].plot(self.metrics[agent_type]['steps'], label=label)
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].legend()
        
        # Bar plot of win rates and completion rates
        metrics = ['win_rate', 'completion_rate']
        x = np.arange(len(metrics))
        width = 0.25
        
        # Plot bars for each agent
        axes[1, 1].bar(x - width, [self.metrics['basic_dqn'][m] for m in metrics], width, label='Basic DQN')
        axes[1, 1].bar(x, [self.metrics['sophisticated_dqn'][m] for m in metrics], width, label='Sophisticated DQN')
        axes[1, 1].bar(x + width, [self.metrics['dueling_dqn'][m] for m in metrics], width, label='Dueling DQN')
        
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Win Rate', 'Completion Rate'])
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('evaluation_results/comparison_plots.png', dpi=100)
        plt.close()
        gc.collect()
