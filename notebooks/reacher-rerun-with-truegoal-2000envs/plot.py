import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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

    # Get the handles and labels from the current legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Create a new legend with only one entry per method type
    new_handles = []
    new_labels = []
    seen_types = set()
    
    # Map of method types to their display names
    type_names = {
        'Baseline': 'Last State',
        'CRL': 'CRL',
        'CRL (Mean Field)': 'CRL',
        'GoalKDE': 'GoalKDE',
        'GoalKDE (Mean Field)': 'GoalKDE',
        'BC': 'BC',
        'BC (Mean Field)': 'BC'
    }
    
    for handle, label in zip(handles, labels):
        display_name = type_names.get(label, label)
        if display_name not in seen_types:
            seen_types.add(display_name)
            new_handles.append(handle)
            new_labels.append(display_name)
    
    # Create new legend with simplified labels
    ax.legend(new_handles, new_labels, loc='best')

    # Save the figure
    plt.savefig(f'goal_distance_comparison_plot_{env_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate goal distance comparison plot')
    parser.add_argument('--env', type=str, required=True, help='Environment name')
    args = parser.parse_args()

    # Generate the goal distance plot
    plot_goal_distance_comparison(args.env)
    print(f"Goal distance comparison plot generated for {args.env} environment")

if __name__ == "__main__":
    main() 