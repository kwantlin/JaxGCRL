import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set global font sizes for better readability
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 26,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 32
})

# Create output directory
output_dir = 'combined_results'
os.makedirs(output_dir, exist_ok=True)

# Read the data for each environment
reacher_data = pd.read_csv('results_reacher/fb_vs_goalkde_last_state_reacher.csv')
ant_data = pd.read_csv('results_ant/fb_vs_goalkde_last_state_ant.csv')
pusher_data = pd.read_csv('results_pusher_easy/fb_vs_goalkde_last_state_pusher_easy.csv')

# Replace "GoalKDE" with "CIRL" in the data and convert to percentages
for data in [reacher_data, ant_data, pusher_data]:
    data['Method Type'] = data['Method Type'].replace('GoalKDE', 'CIRL')
    # Convert from proportions (0-1) to percentages (0-100)
    data['Mean Difference'] = data['Mean Difference'] * 100
    data['Std Error'] = data['Std Error'] * 100

# Create the combined figure
fig, axes = plt.subplots(1, 3, figsize=(24, 10))

# Define colors for consistency
colors = {'FB': '#2ca02c', 'CIRL': '#ff7f0e'}

# Plot for each environment
environments = ['Ant', 'Reacher', 'Pusher']
data_list = [ant_data, reacher_data, pusher_data]

for i, (env_name, data) in enumerate(zip(environments, data_list)):
    ax = axes[i]
    
    # Create the bar plot
    sns.barplot(
        x='Method', 
        y='Mean Difference', 
        hue='Method Type',
        data=data,
        palette=colors,
        ax=ax
    )
    
    # Add error bars
    for j, (_, row) in enumerate(data.iterrows()):
        ax.errorbar(
            j, row['Mean Difference'], 
            yerr=row['Std Error'], 
            fmt='none', 
            color='black', 
            capsize=5
        )
    
    # Customize each subplot
    ax.set_title(f'{env_name}', fontsize=28, fontweight='bold')
    ax.set_ylabel('Imitation Score (%)' if i == 0 else '', fontsize=26)
    ax.set_xlabel('')
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.tick_params(axis='x', rotation=45, labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits to be consistent across all subplots
    ax.set_ylim(0, 100)
    
    # Add legend only to the first subplot
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=24)
    else:
        ax.get_legend().remove()

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/fb_vs_goalkde_last_state_combined.png', dpi=300, bbox_inches='tight')
print(f"Combined figure saved as '{output_dir}/fb_vs_goalkde_last_state_combined.png'")

# Also save as PDF for publication quality
plt.savefig(f'{output_dir}/fb_vs_goalkde_last_state_combined.pdf', bbox_inches='tight')
print(f"Combined figure saved as '{output_dir}/fb_vs_goalkde_last_state_combined.pdf'")

# Create and save summary table
summary_data = []
for env_name, data in [('Reacher', reacher_data), ('Ant', ant_data), ('Pusher', pusher_data)]:
    for _, row in data.iterrows():
        method_short = 'FB' if 'FB' in row['Method'] else 'CIRL'
        summary_data.append({
            'Environment': env_name,
            'Method': method_short,
            'Imitation Score (%)': f"{row['Mean Difference']:.1f} ± {row['Std Error']:.1f}"
        })

# Create DataFrame and pivot for better formatting
summary_df = pd.DataFrame(summary_data)
pivot_df = summary_df.pivot(index='Environment', columns='Method', values='Imitation Score (%)')

# Save the summary table
pivot_df.to_csv(f'{output_dir}/fb_vs_goalkde_summary_table.csv')
print(f"Summary table saved as '{output_dir}/fb_vs_goalkde_summary_table.csv'")

# Also save the raw combined data
combined_data = pd.concat([
    reacher_data.assign(Environment='Reacher'),
    ant_data.assign(Environment='Ant'),
    pusher_data.assign(Environment='Pusher')
], ignore_index=True)
combined_data.to_csv(f'{output_dir}/fb_vs_goalkde_combined_data.csv', index=False)
print(f"Combined data saved as '{output_dir}/fb_vs_goalkde_combined_data.csv'")

plt.show()

# ============================================================================
# Full Trajectory vs Mean Field Combined Plot
# ============================================================================

# Read the data for each environment
reacher_traj_data = pd.read_csv('results_reacher/full_trajectory_vs_mean_field_reacher.csv')
ant_traj_data = pd.read_csv('results_ant/full_trajectory_vs_mean_field_ant.csv')
pusher_traj_data = pd.read_csv('results_pusher_easy/full_trajectory_vs_mean_field_pusher_easy.csv')

# Replace "GoalKDE" with "CIRL" and "BC" with "GCBC" in the data and convert to percentages
for data in [reacher_traj_data, ant_traj_data, pusher_traj_data]:
    data['Method Type'] = data['Method Type'].replace('GoalKDE', 'CIRL')
    data['Method Type'] = data['Method Type'].replace('BC', 'GCBC')
    # Convert from proportions (0-1) to percentages (0-100)
    data['Mean Difference'] = data['Mean Difference'] * 100
    data['Std Error'] = data['Std Error'] * 100

# Create the combined figure
fig, axes = plt.subplots(1, 3, figsize=(26, 11))

# Define colors for consistency
traj_colors = {'CRL': '#1f77b4', 'CIRL': '#ff7f0e', 'GCBC': '#d62728'}

# Plot for each environment
environments = ['Ant', 'Reacher', 'Pusher']
data_list = [ant_traj_data, reacher_traj_data, pusher_traj_data]

for i, (env_name, data) in enumerate(zip(environments, data_list)):
    ax = axes[i]
    
    # Create the bar plot
    sns.barplot(
        x='Method', 
        y='Mean Difference', 
        hue='Method Type',
        data=data,
        palette=traj_colors,
        ax=ax
    )
    
    # Modify bar appearance to distinguish Full Tau vs Mean Field
    bars = ax.patches
    for j, (_, row) in enumerate(data.iterrows()):
        if 'Full Tau' in row['Method']:
            # De-emphasize Full Tau bars with lighter color and white hatching
            bars[j].set_alpha(0.6)  # Make more transparent
            bars[j].set_hatch('////')  # Add diagonal hatching
            bars[j].set_edgecolor('white')  # White hatching lines
            bars[j].set_linewidth(1.5)  # Make lines more visible
        else:  # Mean Field
            # Keep Mean Field bars solid and prominent
            bars[j].set_alpha(1.0)  # Full opacity
    
    # Add error bars
    for j, (_, row) in enumerate(data.iterrows()):
        ax.errorbar(
            j, row['Mean Difference'], 
            yerr=row['Std Error'], 
            fmt='none', 
            color='black', 
            capsize=5
        )
    
    # Customize each subplot
    ax.set_title(f'{env_name}', fontsize=28, fontweight='bold')
    ax.set_ylabel('Imitation Score (%)' if i == 0 else '', fontsize=26)
    ax.set_xlabel('')
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.tick_params(axis='x', rotation=45, labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Each subplot gets its own y-axis limits (no shared y-axis)
    # Let matplotlib auto-scale based on the data
    
    # Add legend only to the first subplot
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        legend1 = ax.legend(handles=handles, labels=labels, loc='upper left', title='Method Type', fontsize=24)
        
        # Add custom legend for Full Tau vs Mean Field distinction
        from matplotlib.patches import Patch
        full_tau_patch = Patch(facecolor='gray', alpha=0.6, hatch='////', 
                              edgecolor='white', linewidth=1.5, label='Full τ')
        mean_field_patch = Patch(facecolor='gray', alpha=1.0, label='Mean Field')
        
        # Add the second legend directly below the first one
        legend2 = ax.legend(handles=[full_tau_patch, mean_field_patch], 
                           loc='upper left', title='Trajectory Type',
                           bbox_to_anchor=(0, 0.8), fontsize=24)
        
        # Add the first legend back
        ax.add_artist(legend1)
    else:
        ax.get_legend().remove()

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/full_trajectory_vs_mean_field_combined.png', dpi=300, bbox_inches='tight')
print(f"Full trajectory vs mean field combined figure saved as '{output_dir}/full_trajectory_vs_mean_field_combined.png'")

# Also save as PDF for publication quality
plt.savefig(f'{output_dir}/full_trajectory_vs_mean_field_combined.pdf', bbox_inches='tight')
print(f"Full trajectory vs mean field combined figure saved as '{output_dir}/full_trajectory_vs_mean_field_combined.pdf'")

# Create and save summary table for full trajectory vs mean field
traj_summary_data = []
for env_name, data in [('Ant', ant_traj_data), ('Reacher', reacher_traj_data), ('Pusher', pusher_traj_data)]:
    for _, row in data.iterrows():
        traj_summary_data.append({
            'Environment': env_name,
            'Method': row['Method'],
            'Method Type': row['Method Type'],
            'Imitation Score (%)': f"{row['Mean Difference']:.1f} ± {row['Std Error']:.1f}"
        })

# Create DataFrame for the trajectory summary
traj_summary_df = pd.DataFrame(traj_summary_data)
traj_summary_df.to_csv(f'{output_dir}/full_trajectory_vs_mean_field_summary_table.csv', index=False)
print(f"Full trajectory vs mean field summary table saved as '{output_dir}/full_trajectory_vs_mean_field_summary_table.csv'")

# Also save the raw combined data
traj_combined_data = pd.concat([
    ant_traj_data.assign(Environment='Ant'),
    reacher_traj_data.assign(Environment='Reacher'),
    pusher_traj_data.assign(Environment='Pusher')
], ignore_index=True)
traj_combined_data.to_csv(f'{output_dir}/full_trajectory_vs_mean_field_combined_data.csv', index=False)
print(f"Full trajectory vs mean field combined data saved as '{output_dir}/full_trajectory_vs_mean_field_combined_data.csv'")

plt.show()

# ============================================================================
# Value of Pretraining Combined Plot
# ============================================================================

# Read the data for each environment
reacher_pretrain_data = pd.read_csv('results_reacher/value_of_pretraining_reacher.csv')
ant_pretrain_data = pd.read_csv('results_ant/value_of_pretraining_ant.csv')
pusher_pretrain_data = pd.read_csv('results_pusher_easy/value_of_pretraining_pusher_easy.csv')

# Replace "GoalKDE" with "CIRL" and "BC" with "GCBC" in the data and convert to percentages
for data in [reacher_pretrain_data, ant_pretrain_data, pusher_pretrain_data]:
    data['Method Type'] = data['Method Type'].replace('GoalKDE', 'CIRL')
    data['Method Type'] = data['Method Type'].replace('BC', 'GCBC')
    # Convert from proportions (0-1) to percentages (0-100)
    data['Mean Difference'] = data['Mean Difference'] * 100
    data['Std Error'] = data['Std Error'] * 100

# Create the combined figure
fig, axes = plt.subplots(1, 3, figsize=(26, 11))

# Define colors for consistency
pretrain_colors = {'GCBC': '#d62728', 'NN': 'purple', 'FB': '#2ca02c', 'CRL': '#1f77b4', 'CIRL': '#ff7f0e'}

# Plot for each environment
environments = ['Ant', 'Reacher', 'Pusher']
data_list = [ant_pretrain_data, reacher_pretrain_data, pusher_pretrain_data]

for i, (env_name, data) in enumerate(zip(environments, data_list)):
    ax = axes[i]
    
    # Create the bar plot
    sns.barplot(
        x='Method', 
        y='Mean Difference', 
        hue='Method Type',
        data=data,
        palette=pretrain_colors,
        ax=ax
    )
    
    # Add error bars
    for j, (_, row) in enumerate(data.iterrows()):
        ax.errorbar(
            j, row['Mean Difference'], 
            yerr=row['Std Error'], 
            fmt='none', 
            color='black', 
            capsize=5
        )
    
    # Customize each subplot
    ax.set_title(f'{env_name}', fontsize=28, fontweight='bold')
    ax.set_ylabel('Imitation Score (%)' if i == 0 else '', fontsize=26)
    ax.set_xlabel('')
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.tick_params(axis='x', rotation=45, labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Each subplot gets its own y-axis limits (no shared y-axis)
    # Let matplotlib auto-scale based on the data
    
    # Add legend only to the first subplot
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=24)
    else:
        ax.get_legend().remove()

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/value_of_pretraining_combined.png', dpi=300, bbox_inches='tight')
print(f"Value of pretraining combined figure saved as '{output_dir}/value_of_pretraining_combined.png'")

# Also save as PDF for publication quality
plt.savefig(f'{output_dir}/value_of_pretraining_combined.pdf', bbox_inches='tight')
print(f"Value of pretraining combined figure saved as '{output_dir}/value_of_pretraining_combined.pdf'")

# Create and save summary table for value of pretraining
pretrain_summary_data = []
for env_name, data in [('Ant', ant_pretrain_data), ('Reacher', reacher_pretrain_data), ('Pusher', pusher_pretrain_data)]:
    for _, row in data.iterrows():
        pretrain_summary_data.append({
            'Environment': env_name,
            'Method': row['Method'],
            'Method Type': row['Method Type'],
            'Imitation Score (%)': f"{row['Mean Difference']:.1f} ± {row['Std Error']:.1f}"
        })

# Create DataFrame for the pretraining summary
pretrain_summary_df = pd.DataFrame(pretrain_summary_data)
pretrain_summary_df.to_csv(f'{output_dir}/value_of_pretraining_summary_table.csv', index=False)
print(f"Value of pretraining summary table saved as '{output_dir}/value_of_pretraining_summary_table.csv'")

# Also save the raw combined data
pretrain_combined_data = pd.concat([
    ant_pretrain_data.assign(Environment='Ant'),
    reacher_pretrain_data.assign(Environment='Reacher'),
    pusher_pretrain_data.assign(Environment='Pusher')
], ignore_index=True)
pretrain_combined_data.to_csv(f'{output_dir}/value_of_pretraining_combined_data.csv', index=False)
print(f"Value of pretraining combined data saved as '{output_dir}/value_of_pretraining_combined_data.csv'")

plt.show()

# ============================================================================
# CRL Oracle vs CRL GoalKDE Combined Plot
# ============================================================================

# Read the data for each environment
reacher_oracle_data = pd.read_csv('results_reacher/crl_oracle_vs_crl_goalkde_reacher.csv')
ant_oracle_data = pd.read_csv('results_ant/crl_oracle_vs_crl_goalkde_ant.csv')
pusher_oracle_data = pd.read_csv('results_pusher_easy/crl_oracle_vs_crl_goalkde_pusher_easy.csv')

# Replace "GoalKDE" with "CIRL" and "CRL" with more descriptive labels in the data and convert to percentages
for data in [reacher_oracle_data, ant_oracle_data, pusher_oracle_data]:
    data['Method Type'] = data['Method Type'].replace('GoalKDE', 'CRL + GoalKDE (CIRL)')
    data['Method Type'] = data['Method Type'].replace('CRL', 'CRL + Oracle Sampling')
    # Convert from proportions (0-1) to percentages (0-100)
    data['Mean Difference'] = data['Mean Difference'] * 100
    data['Std Error'] = data['Std Error'] * 100



# Create the combined figure
fig, axes = plt.subplots(1, 3, figsize=(24, 10))

# Define colors for consistency
oracle_colors = {
    'CRL + Oracle Sampling': '#1f77b4', 
    'CRL + GoalKDE (CIRL)': '#ff7f0e'
}

# Plot for each environment
environments = ['Ant', 'Reacher', 'Pusher']
data_list = [ant_oracle_data, reacher_oracle_data, pusher_oracle_data]

for i, (env_name, data) in enumerate(zip(environments, data_list)):
    ax = axes[i]
    
    # Create the bar plot
    sns.barplot(
        x='Method', 
        y='Mean Difference', 
        hue='Method Type',
        data=data,
        palette=oracle_colors,
        ax=ax
    )
    
    # Add error bars
    for j, (_, row) in enumerate(data.iterrows()):
        ax.errorbar(
            j, row['Mean Difference'], 
            yerr=row['Std Error'], 
            fmt='none', 
            color='black', 
            capsize=5
        )
    
    # Customize each subplot
    ax.set_title(f'{env_name}', fontsize=28, fontweight='bold')
    ax.set_ylabel('Imitation Score (%)' if i == 0 else '', fontsize=26)
    ax.set_xlabel('')
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.tick_params(axis='x', rotation=45, labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits to be consistent across all subplots
    ax.set_ylim(0, 100)
    
    # Add legend only to the first subplot
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=24)
    else:
        ax.get_legend().remove()

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/crl_oracle_vs_crl_goalkde_combined.png', dpi=300, bbox_inches='tight')
print(f"CRL Oracle vs CRL GoalKDE combined figure saved as '{output_dir}/crl_oracle_vs_crl_goalkde_combined.png'")

# Also save as PDF for publication quality
plt.savefig(f'{output_dir}/crl_oracle_vs_crl_goalkde_combined.pdf', bbox_inches='tight')
print(f"CRL Oracle vs CRL GoalKDE combined figure saved as '{output_dir}/crl_oracle_vs_crl_goalkde_combined.pdf'")

# Create and save summary table for CRL Oracle vs CRL GoalKDE
oracle_summary_data = []
for env_name, data in [('Ant', ant_oracle_data), ('Reacher', reacher_oracle_data), ('Pusher', pusher_oracle_data)]:
    for _, row in data.iterrows():
        method_short = 'CRL + Oracle Sampling' if 'Oracle' in row['Method'] else 'CRL + GoalKDE (CIRL)'
        oracle_summary_data.append({
            'Environment': env_name,
            'Method': method_short,
            'Imitation Score (%)': f"{row['Mean Difference']:.1f} ± {row['Std Error']:.1f}"
        })

# Create DataFrame and pivot for better formatting
oracle_summary_df = pd.DataFrame(oracle_summary_data)
oracle_pivot_df = oracle_summary_df.pivot(index='Environment', columns='Method', values='Imitation Score (%)')

# Save the summary table
oracle_pivot_df.to_csv(f'{output_dir}/crl_oracle_vs_crl_goalkde_summary_table.csv')
print(f"CRL Oracle vs CRL GoalKDE summary table saved as '{output_dir}/crl_oracle_vs_crl_goalkde_summary_table.csv'")

# Also save the raw combined data
oracle_combined_data = pd.concat([
    ant_oracle_data.assign(Environment='Ant'),
    reacher_oracle_data.assign(Environment='Reacher'),
    pusher_oracle_data.assign(Environment='Pusher')
], ignore_index=True)
oracle_combined_data.to_csv(f'{output_dir}/crl_oracle_vs_crl_goalkde_combined_data.csv', index=False)
print(f"CRL Oracle vs CRL GoalKDE combined data saved as '{output_dir}/crl_oracle_vs_crl_goalkde_combined_data.csv'")

plt.show()

# ============================================================================
# CRL + GoalKDE Error Analysis Combined Plot
# ============================================================================

# Read the data for each environment
reacher_error_data = pd.read_csv('results_reacher/crl_goalkde_error_analysis_reacher.csv')
ant_error_data = pd.read_csv('results_ant/crl_goalkde_error_analysis_ant.csv')
pusher_error_data = pd.read_csv('results_pusher_easy/crl_goalkde_error_analysis_pusher_easy.csv')

# Replace "GoalKDE" with "CIRL" in the data and convert to percentages
for data in [reacher_error_data, ant_error_data, pusher_error_data]:
    data['Method Type'] = data['Method Type'].replace('CRL + GoalKDE', 'CRL + CIRL')
    # Convert from proportions (0-1) to percentages (0-100)
    data['Mean Difference'] = data['Mean Difference'] * 100
    data['Std Error'] = data['Std Error'] * 100

# Create the combined figure
fig, axes = plt.subplots(1, 3, figsize=(26, 11))

# Define colors for consistency
error_colors = {'CRL + CIRL': '#ff7f0e'}

# Plot for each environment
environments = ['Ant', 'Reacher', 'Pusher']
data_list = [ant_error_data, reacher_error_data, pusher_error_data]

for i, (env_name, data) in enumerate(zip(environments, data_list)):
    ax = axes[i]
    
    # Create the bar plot
    sns.barplot(
        x='Method', 
        y='Mean Difference', 
        hue='Method Type',
        data=data,
        palette=error_colors,
        ax=ax
    )
    
    # Add error bars
    for j, (_, row) in enumerate(data.iterrows()):
        ax.errorbar(
            j, row['Mean Difference'], 
            yerr=row['Std Error'], 
            fmt='none', 
            color='black', 
            capsize=5
        )
    
    # Stacked bar plot elements and dotted lines removed for now
    
    # Customize each subplot
    ax.set_title(f'{env_name}', fontsize=28, fontweight='bold')
    ax.set_ylabel('Imitation Score (%)' if i == 0 else '', fontsize=26)
    ax.set_xlabel('', fontsize=24)
    ax.tick_params(axis='x', rotation=45, labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits to be consistent across all subplots, extending beyond 100 to show error bars
    ax.set_ylim(0, 110)
    
    # Modify the True Goal and Last State bars with different patterns and opacity
    bars = ax.patches
    for j, (_, row) in enumerate(data.iterrows()):
        if row['Method'] == 'True Goal':
            # Use diagonal white lines for True Goal with 70% opacity
            bars[j].set_hatch('////')
            bars[j].set_edgecolor('white')
            bars[j].set_linewidth(2.0)
            bars[j].set_alpha(0.7)
        elif row['Method'] == 'Last State':
            # Use horizontal white lines for Last State with 70% opacity
            bars[j].set_hatch('----')
            bars[j].set_edgecolor('white')
            bars[j].set_linewidth(2.0)
            bars[j].set_alpha(0.7)
    
    # Remove individual legends
    ax.get_legend().remove()

# Custom legend for stacked bars removed since stacked bars are not being plotted

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/crl_goalkde_error_analysis_combined.png', dpi=300, bbox_inches='tight')
print(f"CRL + GoalKDE error analysis combined figure saved as '{output_dir}/crl_goalkde_error_analysis_combined.png'")

# Also save as PDF for publication quality
plt.savefig(f'{output_dir}/crl_goalkde_error_analysis_combined.pdf', bbox_inches='tight')
print(f"CRL + GoalKDE error analysis combined figure saved as '{output_dir}/crl_goalkde_error_analysis_combined.pdf'")

# Create and save summary table for CRL + GoalKDE error analysis
error_summary_data = []
for env_name, data in [('Ant', ant_error_data), ('Reacher', reacher_error_data), ('Pusher', pusher_error_data)]:
    for _, row in data.iterrows():
        error_summary_data.append({
            'Environment': env_name,
            'Method': row['Method'],
            'Method Type': row['Method Type'],
            'Imitation Score (%)': f"{row['Mean Difference']:.1f} ± {row['Std Error']:.1f}"
        })

# Create DataFrame for the error analysis summary
error_summary_df = pd.DataFrame(error_summary_data)
error_summary_df.to_csv(f'{output_dir}/crl_goalkde_error_analysis_summary_table.csv', index=False)
print(f"CRL + GoalKDE error analysis summary table saved as '{output_dir}/crl_goalkde_error_analysis_summary_table.csv'")

# Also save the raw combined data
error_combined_data = pd.concat([
    ant_error_data.assign(Environment='Ant'),
    reacher_error_data.assign(Environment='Reacher'),
    pusher_error_data.assign(Environment='Pusher')
], ignore_index=True)
error_combined_data.to_csv(f'{output_dir}/crl_goalkde_error_analysis_combined_data.csv', index=False)
print(f"CRL + GoalKDE error analysis combined data saved as '{output_dir}/crl_goalkde_error_analysis_combined_data.csv'")

plt.show()
