import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_performance_comparison(env_name):
    # Read the performance comparison data
    performance_df = pd.read_csv(f'performance_comparison_{env_name}.csv')
    
    # Set up the figure
    plt.figure(figsize=(12, 8))

    # Create the bar plot with error bars
    ax = sns.barplot(
        x='Method', 
        y='Mean Difference', 
        hue='Method Type',
        data=performance_df,
        palette=['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue for CRL, Orange for GoalKDE, Green for BC
    )

    # Add error bars
    for i, (_, row) in enumerate(performance_df.iterrows()):
        ax.errorbar(
            i, row['Mean Difference'], 
            yerr=row['Std Error'], 
            fmt='none', 
            color='black', 
            capsize=5
        )

    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='green', linestyle='-', alpha=0.7, label='Zero Regret')

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='best')

    # Customize the plot
    plt.title(f'Regret Compared to Expert Demonstrations ({env_name})', fontsize=16)
    plt.ylabel('Mean Regret', fontsize=14)
    plt.xlabel('Method', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'performance_comparison_plot_{env_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_goal_distance_comparison(env_name):
    # Read the goal distance comparison data
    distance_df = pd.read_csv(f'goal_distance_comparison_{env_name}.csv')
    
    # Set up the figure
    plt.figure(figsize=(12, 6))

    # Create the bar plot for distances
    ax = sns.barplot(
        x='Method', 
        y='Mean Distance', 
        hue='Method Type',
        data=distance_df,
        palette=['gray', '#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e', '#2ca02c', '#2ca02c']
    )

    # Add error bars
    for i, (_, row) in enumerate(distance_df.iterrows()):
        ax.errorbar(
            i, row['Mean Distance'], 
            yerr=row['Std Error'], 
            fmt='none', 
            color='black', 
            capsize=5
        )

    # Customize the plot
    plt.title(f'Goal Distance Comparison ({env_name})', fontsize=16)
    plt.ylabel('Mean Distance to True Goal', fontsize=14)
    plt.xlabel('Method', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='best')

    # Save the figure
    plt.savefig(f'goal_distance_comparison_plot_{env_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate plots from evaluation data')
    parser.add_argument('--env', type=str, required=True, help='Environment name')
    args = parser.parse_args()

    # Read the original data
    performance_df = pd.read_csv(f'performance_comparison_{args.env}.csv')
    distance_df = pd.read_csv(f'goal_distance_comparison_{args.env}.csv')

    # Add true goal reward differences to performance comparison
    true_goal_data = {
        'Method': ['GoalKDE True Goal', 'BC True Goal'],
        'Mean Difference': [
            float(goalkde_reward_diff_true_goal_mean),
            float(bc_reward_diff_true_goal_mean)
        ],
        'Std Error': [
            float(goalkde_reward_diff_true_goal_stderror),
            float(bc_reward_diff_true_goal_stderror)
        ],
        'Method Type': ['GoalKDE', 'BC']
    }
    
    # Append true goal data to performance dataframe
    true_goal_df = pd.DataFrame(true_goal_data)
    performance_df = pd.concat([performance_df, true_goal_df], ignore_index=True)
    
    # Save updated performance comparison data
    performance_df.to_csv(f'performance_comparison_{args.env}.csv', index=False)
    print(f"Updated performance comparison data saved to performance_comparison_{args.env}.csv")

    # Generate both plots
    plot_performance_comparison(args.env)
    plot_goal_distance_comparison(args.env)
    print(f"Plots generated for {args.env} environment")

if __name__ == "__main__":
    main() 