"""
Script to generate detailed tabulated results and statistical comparisons
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

def load_metrics(metrics_path):
    """Load all metrics files"""
    print(f"Loading metrics from: {metrics_path}")
    metrics = {}
    
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics directory not found: {metrics_path}")
        return metrics
    
    files = os.listdir(metrics_path)
    print(f"Found files: {files}")
    
    for filename in files:
        if filename.endswith('_metrics.json'):
            agent_name = filename.split('_metrics')[0]
            file_path = os.path.join(metrics_path, filename)
            print(f"Loading metrics for {agent_name} from {file_path}")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Handle new metrics format
                    if 'episodes' in data:
                        print(f"Found new format metrics for {agent_name}")
                        metrics[agent_name] = {
                            'episode_scores': data['episodes']['score']['values'],
                            'episode_rewards': data['episodes']['reward']['values'],
                            'episode_lengths': data['episodes']['steps']['values']
                        }
                    else:
                        print(f"Found old format metrics for {agent_name}")
                        metrics[agent_name] = data
            except Exception as e:
                print(f"Error loading metrics for {agent_name}: {str(e)}")
    
    print(f"Loaded metrics for agents: {list(metrics.keys())}")
    return metrics

def calculate_statistics(metrics):
    """Calculate detailed statistics for each agent"""
    stats = {}
    
    for agent_name, agent_metrics in metrics.items():
        if agent_name != 'MultiAgent':
            # Basic statistics
            scores = np.array(agent_metrics['episode_scores'])
            rewards = np.array(agent_metrics['episode_rewards'])
            lengths = np.array(agent_metrics['episode_lengths'])
            
            stats[agent_name] = {
                'scores': {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores)
                },
                'rewards': {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'min': np.min(rewards),
                    'max': np.max(rewards),
                    'median': np.median(rewards)
                },
                'lengths': {
                    'mean': np.mean(lengths),
                    'std': np.std(lengths),
                    'min': np.min(lengths),
                    'max': np.max(lengths),
                    'median': np.median(lengths)
                },
                'completion_rate': np.mean(scores > 0) * 100,
                'optimal_rate': np.mean(scores == np.max(scores)) * 100
            }
    
    return stats

def create_performance_table(stats):
    """Create a detailed performance comparison table"""
    # Create DataFrame for easy tabulation
    data = []
    for agent_name, agent_stats in stats.items():
        row = {
            'Agent': agent_name,
            'Avg Score': f"{agent_stats['scores']['mean']:.2f} ± {agent_stats['scores']['std']:.2f}",
            'Max Score': f"{agent_stats['scores']['max']:.0f}",
            'Avg Reward': f"{agent_stats['rewards']['mean']:.2f} ± {agent_stats['rewards']['std']:.2f}",
            'Avg Steps': f"{agent_stats['lengths']['mean']:.2f} ± {agent_stats['lengths']['std']:.2f}",
            'Completion Rate (%)': f"{agent_stats['completion_rate']:.1f}",
            'Optimal Rate (%)': f"{agent_stats['optimal_rate']:.1f}"
        }
        data.append(row)
    
    return pd.DataFrame(data)

def create_episode_analysis(metrics):
    """Create episode-by-episode analysis"""
    if not metrics:
        print("No metrics available for episode analysis")
        return pd.DataFrame()
    
    # Create DataFrame for episode analysis
    data = []
    first_metrics = next(iter(metrics.values()))
    num_episodes = len(first_metrics['episode_scores'])
    
    for episode in range(num_episodes):
        row = {'Episode': episode + 1}
        for agent_name, agent_metrics in metrics.items():
            if agent_name != 'MultiAgent':
                row[f'{agent_name} Score'] = agent_metrics['episode_scores'][episode]
                row[f'{agent_name} Steps'] = agent_metrics['episode_lengths'][episode]
        data.append(row)
    
    return pd.DataFrame(data)

def create_multi_agent_table(metrics):
    """Create multi-agent performance table"""
    if 'MultiAgent' not in metrics:
        return None
    
    multi_metrics = metrics['MultiAgent']
    if isinstance(multi_metrics['episode_scores'][0], list):
        num_agents = len(multi_metrics['episode_scores'][0])
    else:
        num_agents = 1
    
    data = []
    for i in range(num_agents):
        if num_agents > 1:
            scores = [s[i] for s in multi_metrics['episode_scores']]
            rewards = [r[i] for r in multi_metrics['episode_rewards']]
        else:
            scores = multi_metrics['episode_scores']
            rewards = multi_metrics['episode_rewards']
        
        scores = np.array(scores)
        rewards = np.array(rewards)
        
        row = {
            'Agent': f'Agent {i+1}',
            'Avg Score': f"{np.mean(scores):.2f} ± {np.std(scores):.2f}",
            'Max Score': f"{np.max(scores):.0f}",
            'Avg Reward': f"{np.mean(rewards):.2f} ± {np.std(rewards):.2f}",
            'Completion Rate (%)': f"{np.mean(scores > 0) * 100:.1f}",
            'Optimal Rate (%)': f"{np.mean(scores == np.max(scores)) * 100:.1f}"
        }
        data.append(row)
    
    return pd.DataFrame(data)

def save_markdown_report(performance_table, episode_table, multi_agent_table, save_path):
    """Save results as a markdown report"""
    report = ["# Reinforcement Learning Results Analysis\n"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report.append(f"Generated at: {timestamp}\n")
    
    # Overall Performance
    report.append("## Overall Agent Performance\n")
    report.append(performance_table.to_markdown(index=False))
    report.append("\n")
    
    # Episode Analysis
    if not episode_table.empty:
        report.append("## Episode-by-Episode Analysis\n")
        report.append(episode_table.to_markdown(index=False))
        report.append("\n")
    
    # Multi-Agent Results
    if multi_agent_table is not None:
        report.append("## Multi-Agent System Performance\n")
        report.append(multi_agent_table.to_markdown(index=False))
        report.append("\n")
    
    # Statistical Analysis
    report.append("## Statistical Insights\n")
    
    if not performance_table.empty:
        # Best performing agent
        best_agent = performance_table.iloc[performance_table['Avg Score'].str.split('±').str[0].astype(float).idxmax()]
        report.append(f"### Best Performing Agent: {best_agent['Agent']}\n")
        report.extend([
            f"- Average Score: {best_agent['Avg Score']}",
            f"- Maximum Score: {best_agent['Max Score']}",
            f"- Completion Rate: {best_agent['Completion Rate (%)']}%\n"
        ])
    
    # Save report
    report_path = os.path.join(save_path, 'results_analysis.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Saved report to: {report_path}")

def save_csv_results(performance_table, episode_table, multi_agent_table, save_path):
    """Save results as CSV files"""
    if not performance_table.empty:
        path = os.path.join(save_path, 'overall_performance.csv')
        performance_table.to_csv(path, index=False)
        print(f"Saved performance table to: {path}")
    
    if not episode_table.empty:
        path = os.path.join(save_path, 'episode_analysis.csv')
        episode_table.to_csv(path, index=False)
        print(f"Saved episode analysis to: {path}")
    
    if multi_agent_table is not None:
        path = os.path.join(save_path, 'multi_agent_performance.csv')
        multi_agent_table.to_csv(path, index=False)
        print(f"Saved multi-agent analysis to: {path}")

def main():
    # Load metrics
    metrics_path = os.path.join('data', 'metrics')
    metrics = load_metrics(metrics_path)
    
    if not metrics:
        print("No metrics found. Exiting.")
        return
    
    # Calculate statistics
    stats = calculate_statistics(metrics)
    
    # Create tables
    performance_table = create_performance_table(stats)
    episode_table = create_episode_analysis(metrics)
    multi_agent_table = create_multi_agent_table(metrics)
    
    # Save results
    save_path = os.path.join('data', 'reports')
    os.makedirs(save_path, exist_ok=True)
    
    save_markdown_report(performance_table, episode_table, multi_agent_table, save_path)
    save_csv_results(performance_table, episode_table, multi_agent_table, save_path)
    
    print("Results analysis complete! Check the reports directory for detailed tables and analysis.")

if __name__ == "__main__":
    main()
