import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import scienceplots  # noqa: F401
    try:
        plt.style.use(['science', 'ieee'])
    except Exception:
        try:
            plt.style.use(['science', 'ieee', 'no-latex'])
        except Exception:
            pass
except Exception:
    pass
try:
    from palettable.colorbrewer.qualitative import Set2_7  # noqa: F401
    PALETTE_COLORS = Set2_7.mpl_colors
except Exception:
    # Fallback to Set2-like colors
    PALETTE_COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']
import numpy as np
import os

# Set global font sizes for better readability
plt.rcParams.update({
    'font.size': 32,
    'axes.titlesize': 48,
    'axes.labelsize': 42,
    'xtick.labelsize': 38,
    'ytick.labelsize': 38,
    'legend.fontsize': 38,
    'figure.titlesize': 50,
    'hatch.linewidth': 3.0
})

# Global, consistent color map for method types across all figures
METHOD_TYPE_ORDER = ['GCBC', 'NN', 'FB', 'CRL', 'CIRL', 'CRL + Oracle', 'HILP']
METHOD_TYPE_COLOR = {t: PALETTE_COLORS[i % len(PALETTE_COLORS)] for i, t in enumerate(METHOD_TYPE_ORDER)}

# Swap colors for NN and CIRL as requested
_nn_color = METHOD_TYPE_COLOR.get('NN')
_cirl_color = METHOD_TYPE_COLOR.get('CIRL')
if _nn_color is not None and _cirl_color is not None:
    METHOD_TYPE_COLOR['NN'], METHOD_TYPE_COLOR['CIRL'] = _cirl_color, _nn_color

# Reassign CRL + Oracle to a less distracting Set2 color (avoid bright yellow)
try:
    METHOD_TYPE_COLOR['CRL + Oracle'] = PALETTE_COLORS[6]
except Exception:
    # Fallback to a non-yellow Set2 color if palette shorter
    METHOD_TYPE_COLOR['CRL + Oracle'] = '#e5c494'
# Ensure HILP has a distinct color from CRL + Oracle
METHOD_TYPE_COLOR['HILP'] = '#9467bd'

# Aliases to ensure semantically equivalent labels share the same color
METHOD_TYPE_ALIASES = {
    'CRL + GoalKDE (CIRL)': 'CIRL',
    'CRL + CIRL': 'CIRL',
    'FB (Offline)': 'FB',
}

def get_color_for_type(method_type: str) -> str:
    canonical = METHOD_TYPE_ALIASES.get(method_type, method_type)
    return METHOD_TYPE_COLOR.get(canonical, '#888888')

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

# Single-axis combined plot with shared y-axis and environment x-ticks
fig, ax = plt.subplots(figsize=(14, 8))

# Define colors for consistency by Method Type using global map
colors = {'FB': get_color_for_type('FB'), 'CIRL': get_color_for_type('CIRL')}

# Prepare combined data with Environment column
ant_plot = ant_data.copy().assign(Environment='Ant')
reacher_plot = reacher_data.copy().assign(Environment='Reacher')
pusher_plot = pusher_data.copy().assign(Environment='Pusher')
combined_plot = pd.concat([ant_plot, reacher_plot, pusher_plot], ignore_index=True)

# Determine methods per environment (union across all)
environments = ['Reacher', 'Pusher', 'Ant']

# Map each Method to its Method Type (first occurrence) and color
method_to_type = combined_plot.dropna(subset=['Method']).drop_duplicates('Method').set_index('Method')['Method Type'].to_dict()

# Order methods with CIRL last
def _is_cirl_type(t):
    return (t == 'CIRL') or (isinstance(t, str) and 'CIRL' in t)
all_methods = combined_plot['Method'].unique().tolist()
methods = sorted(all_methods, key=lambda m: (1 if _is_cirl_type(method_to_type.get(m, '')) else 0, str(m)))

method_palette = {m: colors.get(method_to_type.get(m, ''), '#888888') for m in methods}

# Compute positions
env_positions = np.arange(len(environments))
max_methods_per_env = max((combined_plot[combined_plot['Environment'] == env]['Method'].nunique() for env in environments))
group_width = 0.8
bar_width = group_width / max_methods_per_env if max_methods_per_env > 0 else 0.4

# For consistent order of methods within each environment
for i, env in enumerate(environments):
    env_df = combined_plot[combined_plot['Environment'] == env]
    env_methods = [m for m in methods if m in env_df['Method'].values]
    num_env_methods = len(env_methods)
    if num_env_methods == 0:
        continue
    # Center bars around env position
    offsets = (np.arange(num_env_methods) - (num_env_methods - 1) / 2.0) * bar_width
    for offset, m in zip(offsets, env_methods):
        row = env_df[env_df['Method'] == m].iloc[0]
        height = row['Mean Difference']
        err = row['Std Error']
        xpos = env_positions[i] + offset
        ax.bar(xpos, height, width=bar_width * 0.9, color=method_palette[m], edgecolor='none', linewidth=0, label=m if i == 0 else None)
        ax.errorbar(xpos, height, yerr=err, fmt='none', ecolor='black', capsize=5, linewidth=1.0)

# X-axis formatting
ax.set_xticks(env_positions)
ax.set_xticklabels(environments, fontsize=38, rotation=0)
ax.set_xlabel('')

# Y-axis formatting with ~3 major ticks and ~5 minor ticks
from matplotlib.ticker import FixedLocator, AutoMinorLocator
ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_ylabel('Imitation Score (\\%)', fontsize=42)
ax.tick_params(axis='y', labelsize=38)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.7)
ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3)

# Y limits with headroom for error bars
if not combined_plot.empty:
    top_val = float(np.nanmax(combined_plot['Mean Difference'] + combined_plot['Std Error']))
else:
    top_val = 100.0
pad = 0.05 * max(1.0, top_val)
ymax = max(100.0, top_val + pad)
ymax = float(int(np.ceil(ymax / 10.0)) * 10)
ax.set_ylim(0, ymax)

# Legend by Method Type (unique types)
from matplotlib.patches import Patch
type_to_color = {}
for m, t in method_to_type.items():
    if t in colors and t not in type_to_color:
        type_to_color[t] = colors[t]
ordered_types = sorted(type_to_color.keys(), key=lambda t: (1 if _is_cirl_type(t) else 0, str(t)))
legend_handles = [Patch(facecolor=type_to_color[t], edgecolor='none', label=t) for t in ordered_types]
if legend_handles:
    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_legend.remove()
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        bbox_transform=ax.transAxes,
        fontsize=38,
        title=None,
        borderaxespad=0.0,
        frameon=True
    )

plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/fb_vs_goalkde_last_state_combined.png', dpi=300, bbox_inches='tight')
print(f"Combined figure saved as '{output_dir}/fb_vs_goalkde_last_state_combined.png'")

# Also save as PDF for publication quality
plt.savefig(f'{output_dir}/fb_vs_goalkde_last_state_combined.pdf', bbox_inches='tight')
print(f"Combined figure saved as '{output_dir}/fb_vs_goalkde_last_state_combined.pdf'")

# Create and save summary table (ordered: Reacher, Pusher, Ant)
summary_data = []
for env_name, data in [('Reacher', reacher_data), ('Pusher', pusher_data), ('Ant', ant_data)]:
    for _, row in data.iterrows():
        m = str(row['Method'])
        if 'FB (Offline' in m:
            method_short = 'FB (Offline)'
        elif m.startswith('FB'):
            method_short = 'FB'
        else:
            method_short = 'CIRL'
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

# Single-axis combined plot for Full Trajectory vs Mean Field
fig, ax = plt.subplots(figsize=(14, 8))

# Define colors for consistency using global map
traj_colors = {'CRL': get_color_for_type('CRL'), 'CIRL': get_color_for_type('CIRL'), 'GCBC': get_color_for_type('GCBC')}

# Prepare combined data with Environment column
ant_traj_plot = ant_traj_data.copy().assign(Environment='Ant')
reacher_traj_plot = reacher_traj_data.copy().assign(Environment='Reacher')
pusher_traj_plot = pusher_traj_data.copy().assign(Environment='Pusher')
traj_combined_plot = pd.concat([ant_traj_plot, reacher_traj_plot, pusher_traj_plot], ignore_index=True)

environments = ['Reacher', 'Pusher', 'Ant']
all_methods = traj_combined_plot['Method'].unique().tolist()
method_to_type = traj_combined_plot.dropna(subset=['Method']).drop_duplicates('Method').set_index('Method')['Method Type'].to_dict()
def _is_cirl_type(t):
    return (t == 'CIRL') or (isinstance(t, str) and 'CIRL' in t)
methods = sorted(all_methods, key=lambda m: (1 if _is_cirl_type(method_to_type.get(m, '')) else 0, str(m)))
method_palette = {m: traj_colors.get(method_to_type.get(m, ''), '#888888') for m in methods}

env_positions = np.arange(len(environments))
max_methods_per_env = max((traj_combined_plot[traj_combined_plot['Environment'] == env]['Method'].nunique() for env in environments))
group_width = 0.8
bar_width = group_width / max_methods_per_env if max_methods_per_env > 0 else 0.4

for i, env in enumerate(environments):
    env_df = traj_combined_plot[traj_combined_plot['Environment'] == env]
    env_methods = [m for m in methods if m in env_df['Method'].values]
    num_env_methods = len(env_methods)
    if num_env_methods == 0:
        continue
    offsets = (np.arange(num_env_methods) - (num_env_methods - 1) / 2.0) * bar_width
    for offset, m in zip(offsets, env_methods):
        row = env_df[env_df['Method'] == m].iloc[0]
        height = row['Mean Difference']
        err = row['Std Error']
        xpos = env_positions[i] + offset
        bar = ax.bar(xpos, height, width=bar_width * 0.9, color=method_palette[m], edgecolor='none', linewidth=0)
        # Style bars for Full Tau vs Mean Field
        if 'Full Tau' in m:
            bar[0].set_alpha(0.6)
            bar[0].set_hatch('////')
            bar[0].set_edgecolor('white')
            bar[0].set_linewidth(2.0)
        else:
            bar[0].set_alpha(1.0)
        ax.errorbar(xpos, height, yerr=err, fmt='none', ecolor='black', capsize=5, linewidth=1.0)

ax.set_xticks(env_positions)
ax.set_xticklabels(environments, fontsize=38, rotation=0)
ax.set_xlabel('')

from matplotlib.ticker import FixedLocator, AutoMinorLocator
ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_ylabel('Imitation Score (\\%)', fontsize=42)
ax.tick_params(axis='y', labelsize=38)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.7)
ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3)

# Y limits with headroom for error bars
if not traj_combined_plot.empty:
    top_val = float(np.nanmax(traj_combined_plot['Mean Difference'] + traj_combined_plot['Std Error']))
else:
    top_val = 100.0
pad = 0.05 * max(1.0, top_val)
ymax = max(100.0, top_val + pad)
ymax = float(int(np.ceil(ymax / 10.0)) * 10)
ax.set_ylim(0, ymax)

from matplotlib.patches import Patch
type_to_color = {}
for m, t in method_to_type.items():
    if t in traj_colors and t not in type_to_color:
        type_to_color[t] = traj_colors[t]
ordered_types = sorted(type_to_color.keys(), key=lambda t: (1 if _is_cirl_type(t) else 0, str(t)))
legend_handles = [Patch(facecolor=type_to_color[t], edgecolor='none', label=t) for t in ordered_types]
if legend_handles:
    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_legend.remove()
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        bbox_transform=ax.transAxes,
        fontsize=38,
        title=None,
        borderaxespad=0.0,
        frameon=True
    )

plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/full_trajectory_vs_mean_field_combined.png', dpi=300, bbox_inches='tight')
print(f"Full trajectory vs mean field combined figure saved as '{output_dir}/full_trajectory_vs_mean_field_combined.png'")

# Also save as PDF for publication quality
plt.savefig(f'{output_dir}/full_trajectory_vs_mean_field_combined.pdf', bbox_inches='tight')
print(f"Full trajectory vs mean field combined figure saved as '{output_dir}/full_trajectory_vs_mean_field_combined.pdf'")

# Create and save summary table for full trajectory vs mean field
traj_summary_data = []
for env_name, data in [('Reacher', reacher_traj_data), ('Pusher', pusher_traj_data), ('Ant', ant_traj_data)]:
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

# Single-axis combined plot for Value of Pretraining
fig, ax = plt.subplots(figsize=(14, 8))

pretrain_colors = {
    'GCBC': get_color_for_type('GCBC'),
    'NN': get_color_for_type('NN'),
    'FB': get_color_for_type('FB'),
    'CRL': get_color_for_type('CRL'),
    'CIRL': get_color_for_type('CIRL'),
    'HILP': get_color_for_type('HILP'),
}

ant_pre_plot = ant_pretrain_data.copy().assign(Environment='Ant')
reacher_pre_plot = reacher_pretrain_data.copy().assign(Environment='Reacher')
pusher_pre_plot = pusher_pretrain_data.copy().assign(Environment='Pusher')
pre_combined_plot = pd.concat([ant_pre_plot, reacher_pre_plot, pusher_pre_plot], ignore_index=True)

# Remove GCBC and PSM from the plot as requested
pre_combined_plot = pre_combined_plot[~pre_combined_plot['Method Type'].isin(['GCBC', 'PSM'])]

environments = ['Reacher', 'Pusher', 'Ant']
method_to_type = pre_combined_plot.dropna(subset=['Method']).drop_duplicates('Method').set_index('Method')['Method Type'].to_dict()
# Order methods by desired Method Type order
type_order = ['GCBC', 'NN', 'FB', 'HILP', 'CRL', 'CIRL']
all_methods = pre_combined_plot['Method'].unique().tolist()
def _method_sort_key(m):
    t = method_to_type.get(m, '')
    return (type_order.index(t) if t in type_order else len(type_order), str(m))
methods = sorted(all_methods, key=_method_sort_key)
method_palette = {m: get_color_for_type(method_to_type.get(m, '')) for m in methods}

env_positions = np.arange(len(environments))
max_methods_per_env = max((pre_combined_plot[pre_combined_plot['Environment'] == env]['Method'].nunique() for env in environments))
group_width = 0.8
bar_width = group_width / max_methods_per_env if max_methods_per_env > 0 else 0.4

for i, env in enumerate(environments):
    env_df = pre_combined_plot[pre_combined_plot['Environment'] == env]
    env_methods = [m for m in methods if m in env_df['Method'].values]
    num_env_methods = len(env_methods)
    if num_env_methods == 0:
        continue
    offsets = (np.arange(num_env_methods) - (num_env_methods - 1) / 2.0) * bar_width
    for offset, m in zip(offsets, env_methods):
        row = env_df[env_df['Method'] == m].iloc[0]
        height = row['Mean Difference']
        err = row['Std Error']
        xpos = env_positions[i] + offset
        # Draw bar; if FB (Offline) has zero height, render a thin visible bar
        draw_height = height
        is_fb_offline = isinstance(m, str) and 'FB (Offline' in m
        if is_fb_offline and (not np.isfinite(draw_height) or draw_height <= 0.0):
            draw_height = 0.5  # small visible height in percent units for clarity
        bar = ax.bar(xpos, draw_height, width=bar_width * 0.9, color=method_palette[m], edgecolor='none', linewidth=0)
        # Visually distinguish FB (Offline) with white diagonal hatch while keeping FB color
        if is_fb_offline:
            bar[0].set_hatch('////')
            bar[0].set_edgecolor('white')
            bar[0].set_linewidth(2.0)
        ax.errorbar(xpos, height, yerr=err, fmt='none', ecolor='black', capsize=5, linewidth=1.0)

ax.set_xticks(env_positions)
ax.set_xticklabels(environments, fontsize=38, rotation=0)
ax.set_xlabel('')

from matplotlib.ticker import FixedLocator, AutoMinorLocator
ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_ylabel('Imitation Score (\\%)', fontsize=42)
ax.tick_params(axis='y', labelsize=38)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.7)
ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3)

# Y limits with headroom for error bars
if not pre_combined_plot.empty:
    top_val = float(np.nanmax(pre_combined_plot['Mean Difference'] + pre_combined_plot['Std Error']))
else:
    top_val = 100.0
pad = 0.05 * max(1.0, top_val)
ymax = max(100.0, top_val + pad)
ymax = float(int(np.ceil(ymax / 10.0)) * 10)
ax.set_ylim(0, ymax)

from matplotlib.patches import Patch
type_to_color = {t: get_color_for_type(t) for t in set(method_to_type.values())}
# Desired legend order: two columns x three rows
desired_order = ['NN', 'FB', 'FB (Offline)', 'HILP', 'CIRL']
# Determine if FB (Offline) is present in the data
_has_fb_offline = any(isinstance(m, str) and 'FB (Offline' in m for m in pre_combined_plot['Method'].unique())
# Build handles in the desired order, skipping entries not present in the data
legend_handles = []
for label in desired_order:
    if label == 'FB (Offline)':
        if _has_fb_offline:
            legend_handles.append(Patch(facecolor=get_color_for_type('FB'),
                                        edgecolor='white',
                                        linewidth=2.0,
                                        hatch='////',
                                        label='FB (Offline)'))
    else:
        if label in type_to_color:
            legend_handles.append(Patch(facecolor=type_to_color[label],
                                        edgecolor='none',
                                        label=label))
if legend_handles:
    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_legend.remove()
    ax.legend(handles=legend_handles,
              loc='upper right',
              bbox_to_anchor=(0.99, 0.99),
              bbox_transform=ax.transAxes,
              fontsize=38,
              title=None,
              borderaxespad=0.0,
              frameon=True,
              ncol=2,
              columnspacing=0.25,
              handletextpad=0.3,
              labelspacing=0.2,
              handlelength=1.2)

plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/value_of_pretraining_combined.png', dpi=300, bbox_inches='tight')
print(f"Value of pretraining combined figure saved as '{output_dir}/value_of_pretraining_combined.png'")

# Also save as PDF for publication quality
plt.savefig(f'{output_dir}/value_of_pretraining_combined.pdf', bbox_inches='tight')
print(f"Value of pretraining combined figure saved as '{output_dir}/value_of_pretraining_combined.pdf'")

# Create and save summary table for value of pretraining
pretrain_summary_data = []
for env_name, data in [('Reacher', reacher_pretrain_data), ('Pusher', pusher_pretrain_data), ('Ant', ant_pretrain_data)]:
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
    data['Method Type'] = data['Method Type'].replace('CRL', 'CRL + Oracle')
    # Convert from proportions (0-1) to percentages (0-100)
    data['Mean Difference'] = data['Mean Difference'] * 100
    data['Std Error'] = data['Std Error'] * 100



# Single-axis combined plot for CRL Oracle vs CRL GoalKDE
fig, ax = plt.subplots(figsize=(14, 8))

oracle_colors = {
    'CRL + Oracle': get_color_for_type('CRL + Oracle'),
    'CRL + GoalKDE (CIRL)': get_color_for_type('CIRL')
}

ant_oracle_plot = ant_oracle_data.copy().assign(Environment='Ant')
reacher_oracle_plot = reacher_oracle_data.copy().assign(Environment='Reacher')
pusher_oracle_plot = pusher_oracle_data.copy().assign(Environment='Pusher')
oracle_combined_plot_ax = pd.concat([ant_oracle_plot, reacher_oracle_plot, pusher_oracle_plot], ignore_index=True)

environments = ['Reacher', 'Pusher', 'Ant']
all_methods = oracle_combined_plot_ax['Method'].unique().tolist()
method_to_type = oracle_combined_plot_ax.dropna(subset=['Method']).drop_duplicates('Method').set_index('Method')['Method Type'].to_dict()
def _is_cirl_type(t):
    return (t == 'CIRL') or (isinstance(t, str) and 'CIRL' in t)
methods = sorted(all_methods, key=lambda m: (1 if _is_cirl_type(method_to_type.get(m, '')) else 0, str(m)))
method_palette = {m: oracle_colors.get(method_to_type.get(m, ''), '#888888') for m in methods}

env_positions = np.arange(len(environments))
max_methods_per_env = max((oracle_combined_plot_ax[oracle_combined_plot_ax['Environment'] == env]['Method'].nunique() for env in environments))
group_width = 0.8
bar_width = group_width / max_methods_per_env if max_methods_per_env > 0 else 0.4

for i, env in enumerate(environments):
    env_df = oracle_combined_plot_ax[oracle_combined_plot_ax['Environment'] == env]
    env_methods = [m for m in methods if m in env_df['Method'].values]
    num_env_methods = len(env_methods)
    if num_env_methods == 0:
        continue
    offsets = (np.arange(num_env_methods) - (num_env_methods - 1) / 2.0) * bar_width
    for offset, m in zip(offsets, env_methods):
        row = env_df[env_df['Method'] == m].iloc[0]
        height = row['Mean Difference']
        err = row['Std Error']
        xpos = env_positions[i] + offset
        ax.bar(xpos, height, width=bar_width * 0.9, color=method_palette[m], edgecolor='none', linewidth=0)
        ax.errorbar(xpos, height, yerr=err, fmt='none', ecolor='black', capsize=5, linewidth=1.0)

ax.set_xticks(env_positions)
ax.set_xticklabels(environments, fontsize=38, rotation=0)
ax.set_xlabel('')

from matplotlib.ticker import FixedLocator, AutoMinorLocator
ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_ylabel('Imitation Score (\\%)', fontsize=42)
ax.tick_params(axis='y', labelsize=38)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.7)
ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3)

# Y limits with headroom for error bars
if not oracle_combined_plot_ax.empty:
    top_val = float(np.nanmax(oracle_combined_plot_ax['Mean Difference'] + oracle_combined_plot_ax['Std Error']))
else:
    top_val = 100.0
pad = 0.05 * max(1.0, top_val)
ymax = max(100.0, top_val + pad)
ymax = float(int(np.ceil(ymax / 10.0)) * 10)
ax.set_ylim(0, ymax)

from matplotlib.patches import Patch
type_to_color = {}
for m, t in method_to_type.items():
    if t in oracle_colors and t not in type_to_color:
        type_to_color[t] = oracle_colors[t]
ordered_types = sorted(type_to_color.keys(), key=lambda t: (1 if _is_cirl_type(t) else 0, str(t)))
legend_handles = [Patch(facecolor=type_to_color[t], edgecolor='none', label=t) for t in ordered_types]
if legend_handles:
    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_legend.remove()
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.995),
        bbox_transform=ax.transAxes,
        fontsize=38,
        title=None,
        borderaxespad=0.0,
        frameon=False
    )

plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/crl_oracle_vs_crl_goalkde_combined.png', dpi=300, bbox_inches='tight')
print(f"CRL Oracle vs CRL GoalKDE combined figure saved as '{output_dir}/crl_oracle_vs_crl_goalkde_combined.png'")

# Also save as PDF for publication quality
plt.savefig(f'{output_dir}/crl_oracle_vs_crl_goalkde_combined.pdf', bbox_inches='tight')
print(f"CRL Oracle vs CRL GoalKDE combined figure saved as '{output_dir}/crl_oracle_vs_crl_goalkde_combined.pdf'")

# Create and save summary table for CRL Oracle vs CRL GoalKDE
oracle_summary_data = []
for env_name, data in [('Reacher', reacher_oracle_data), ('Pusher', pusher_oracle_data), ('Ant', ant_oracle_data)]:
    for _, row in data.iterrows():
        method_short = 'CRL + Oracle' if 'Oracle' in row['Method'] else 'CRL + GoalKDE (CIRL)'
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

# Single-axis combined plot for CRL + GoalKDE Error Analysis
fig, ax = plt.subplots(figsize=(14, 8))

error_colors = {'CRL + CIRL': get_color_for_type('CIRL')}

ant_err_plot = ant_error_data.copy().assign(Environment='Ant')
reacher_err_plot = reacher_error_data.copy().assign(Environment='Reacher')
pusher_err_plot = pusher_error_data.copy().assign(Environment='Pusher')
err_combined_plot = pd.concat([ant_err_plot, reacher_err_plot, pusher_err_plot], ignore_index=True)

environments = ['Reacher', 'Pusher', 'Ant']
all_methods = err_combined_plot['Method'].unique().tolist()
method_to_type = err_combined_plot.dropna(subset=['Method']).drop_duplicates('Method').set_index('Method')['Method Type'].to_dict()

# Order so baseline bars first and CRL + CIRL last
baseline_order = ['True Goal', 'Last State']
def _is_cirl_method(m):
    t = method_to_type.get(m, '')
    return (t == 'CIRL') or (isinstance(t, str) and 'CIRL' in t) or (isinstance(m, str) and 'CIRL' in m)
non_baseline = [m for m in all_methods if m not in baseline_order]
non_baseline_sorted = sorted(non_baseline, key=lambda m: (1 if _is_cirl_method(m) else 0, str(m)))
methods = [m for m in baseline_order if m in all_methods] + non_baseline_sorted
method_palette = {m: error_colors.get(method_to_type.get(m, ''), '#888888') for m in methods}

env_positions = np.arange(len(environments))
max_methods_per_env = max((err_combined_plot[err_combined_plot['Environment'] == env]['Method'].nunique() for env in environments))
group_width = 0.8
bar_width = group_width / max_methods_per_env if max_methods_per_env > 0 else 0.4

for i, env in enumerate(environments):
    env_df = err_combined_plot[err_combined_plot['Environment'] == env]
    env_methods = [m for m in methods if m in env_df['Method'].values]
    num_env_methods = len(env_methods)
    if num_env_methods == 0:
        continue
    offsets = (np.arange(num_env_methods) - (num_env_methods - 1) / 2.0) * bar_width
    for offset, m in zip(offsets, env_methods):
        row = env_df[env_df['Method'] == m].iloc[0]
        height = row['Mean Difference']
        err = row['Std Error']
        xpos = env_positions[i] + offset
        bar = ax.bar(xpos, height, width=bar_width * 0.9, color=method_palette[m], edgecolor='none', linewidth=0)
        # Pattern for True Goal vs Last State
        if m == 'True Goal':
            bar[0].set_hatch('/')
            bar[0].set_edgecolor('white')
            bar[0].set_linewidth(2.0)
            bar[0].set_alpha(0.7)
        elif m == 'Last State':
            bar[0].set_hatch('o')
            bar[0].set_edgecolor('white')
            bar[0].set_linewidth(2.0)
            bar[0].set_alpha(0.7)
        ax.errorbar(xpos, height, yerr=err, fmt='none', ecolor='black', capsize=5, linewidth=1.0)

ax.set_xticks(env_positions)
ax.set_xticklabels(environments, fontsize=34, rotation=0)
ax.set_xlabel('', fontsize=34)

from matplotlib.ticker import FixedLocator, AutoMinorLocator
ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_ylabel('Imitation Score (\\%)', fontsize=36)
ax.tick_params(axis='y', labelsize=34)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.7)
ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3)

# Y limits with headroom for error bars
if not err_combined_plot.empty:
    top_val = float(np.nanmax(err_combined_plot['Mean Difference'] + err_combined_plot['Std Error']))
else:
    top_val = 100.0
pad = 0.05 * max(1.0, top_val)
ymax = max(100.0, top_val + pad)
ymax = float(int(np.ceil(ymax / 10.0)) * 10)
ax.set_ylim(0, ymax)

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=error_colors['CRL + CIRL'], edgecolor='white', linewidth=2.0, hatch='/', alpha=0.7, label='True Goal'),
    Patch(facecolor=error_colors['CRL + CIRL'], edgecolor='white', linewidth=2.0, hatch='o', alpha=0.7, label='Last State'),
    Patch(facecolor=error_colors['CRL + CIRL'], edgecolor='none', linewidth=1.0, label='CIRL')
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=34, title=None)

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


# ============================================================================
# CRL Oracle vs CRL GoalKDE Combined Plot on Simple Mazes
# ============================================================================
try:
    plt.rcParams['text.usetex'] = False
except Exception:
    pass
# Read the data for each environment
u_maze_oracle_data = pd.read_csv('results_simple_u_maze/crl_oracle_vs_crl_goalkde_simple_u_maze.csv')
big_maze_oracle_data = pd.read_csv('results_simple_big_maze/crl_oracle_vs_crl_goalkde_simple_big_maze.csv')
hardest_maze_oracle_data = pd.read_csv('results_simple_hardest_maze/crl_oracle_vs_crl_goalkde_simple_hardest_maze.csv')

# Replace "GoalKDE" with "CIRL" and "CRL" with more descriptive labels in the data and convert to percentages
for data in [u_maze_oracle_data, big_maze_oracle_data, hardest_maze_oracle_data]:
    data['Method Type'] = data['Method Type'].replace('GoalKDE', 'CRL + GoalKDE (CIRL)')
    data['Method Type'] = data['Method Type'].replace('CRL', 'CRL + Oracle')
    # Convert from proportions (0-1) to percentages (0-100)
    data['Mean Difference'] = data['Mean Difference'] * 100
    data['Std Error'] = data['Std Error'] * 100



# Single-axis combined plot for CRL Oracle vs CRL GoalKDE
fig, ax = plt.subplots(figsize=(14, 8))


oracle_colors = {
    'CRL + Oracle': get_color_for_type('CRL + Oracle'),
    'CRL + GoalKDE (CIRL)': get_color_for_type('CIRL')
}

u_maze_oracle_plot = u_maze_oracle_data.copy().assign(Environment='U-Maze')
big_maze_oracle_plot = big_maze_oracle_data.copy().assign(Environment='Big Maze')
hardest_maze_oracle_plot = hardest_maze_oracle_data.copy().assign(Environment='Hardest Maze')
oracle_combined_plot_ax = pd.concat([u_maze_oracle_plot, big_maze_oracle_plot, hardest_maze_oracle_plot], ignore_index=True)

environments = ['U-Maze', 'Big Maze', 'Hardest Maze']
all_methods = oracle_combined_plot_ax['Method'].unique().tolist()
method_to_type = oracle_combined_plot_ax.dropna(subset=['Method']).drop_duplicates('Method').set_index('Method')['Method Type'].to_dict()
def _is_cirl_type(t):
    return (t == 'CIRL') or (isinstance(t, str) and 'CIRL' in t)
methods = sorted(all_methods, key=lambda m: (1 if _is_cirl_type(method_to_type.get(m, '')) else 0, str(m)))
method_palette = {m: oracle_colors.get(method_to_type.get(m, ''), '#888888') for m in methods}

env_positions = np.arange(len(environments))
max_methods_per_env = max((oracle_combined_plot_ax[oracle_combined_plot_ax['Environment'] == env]['Method'].nunique() for env in environments))
group_width = 0.8
bar_width = group_width / max_methods_per_env if max_methods_per_env > 0 else 0.4

for i, env in enumerate(environments):
    env_df = oracle_combined_plot_ax[oracle_combined_plot_ax['Environment'] == env]
    env_methods = [m for m in methods if m in env_df['Method'].values]
    num_env_methods = len(env_methods)
    if num_env_methods == 0:
        continue
    offsets = (np.arange(num_env_methods) - (num_env_methods - 1) / 2.0) * bar_width
    for offset, m in zip(offsets, env_methods):
        row = env_df[env_df['Method'] == m].iloc[0]
        height = row['Mean Difference']
        err = row['Std Error']
        xpos = env_positions[i] + offset
        ax.bar(xpos, height, width=bar_width * 0.9, color=method_palette[m], edgecolor='none', linewidth=0)
        ax.errorbar(xpos, height, yerr=err, fmt='none', ecolor='black', capsize=5, linewidth=1.0)

ax.set_xticks(env_positions)
ax.set_xticklabels(environments, fontsize=38, rotation=0)
ax.set_xlabel('')

from matplotlib.ticker import FixedLocator, AutoMinorLocator
ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.set_ylabel('Imitation Score (%)', fontsize=42)
ax.tick_params(axis='y', labelsize=38)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.7)
ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3)

# Y limits with headroom for error bars
if not oracle_combined_plot_ax.empty:
    top_val = float(np.nanmax(oracle_combined_plot_ax['Mean Difference'] + oracle_combined_plot_ax['Std Error']))
else:
    top_val = 100.0
pad = 0.05 * max(1.0, top_val)
ymax = max(100.0, top_val + pad)
ymax = float(int(np.ceil(ymax / 10.0)) * 10)
ax.set_ylim(0, ymax)

from matplotlib.patches import Patch
type_to_color = {}
for m, t in method_to_type.items():
    if t in oracle_colors and t not in type_to_color:
        type_to_color[t] = oracle_colors[t]
ordered_types = sorted(type_to_color.keys(), key=lambda t: (1 if _is_cirl_type(t) else 0, str(t)))
legend_handles = [Patch(facecolor=type_to_color[t], edgecolor='none', label=t) for t in ordered_types]
if legend_handles:
    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_legend.remove()
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.995),
        bbox_transform=ax.transAxes,
        fontsize=38,
        title=None,
        borderaxespad=0.0,
        frameon=False
    )

plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/crl_oracle_vs_crl_goalkde_simple_mazes_combined.png', dpi=300, bbox_inches='tight')
print(f"CRL Oracle vs CRL GoalKDE combined figure saved as '{output_dir}/crl_oracle_vs_crl_goalkde_simple_mazes_combined.png'")

# Also save as PDF for publication quality
plt.savefig(f'{output_dir}/crl_oracle_vs_crl_goalkde_simple_mazes_combined.pdf', bbox_inches='tight')
print(f"CRL Oracle vs CRL GoalKDE combined figure saved as '{output_dir}/crl_oracle_vs_crl_goalkde_simple_mazes_combined.pdf'")

# Create and save summary table for CRL Oracle vs CRL GoalKDE
oracle_summary_data = []
for env_name, data in [('U-Maze', u_maze_oracle_data), ('Big Maze', big_maze_oracle_data), ('Hardest Maze', hardest_maze_oracle_data)]:
    for _, row in data.iterrows():
        method_short = 'CRL + Oracle' if 'Oracle' in row['Method'] else 'CRL + GoalKDE (CIRL)'
        oracle_summary_data.append({
            'Environment': env_name,
            'Method': method_short,
            'Imitation Score (%)': f"{row['Mean Difference']:.1f} ± {row['Std Error']:.1f}"
        })

# Create DataFrame and pivot for better formatting
oracle_summary_df = pd.DataFrame(oracle_summary_data)
oracle_pivot_df = oracle_summary_df.pivot(index='Environment', columns='Method', values='Imitation Score (%)')

# Save the summary table
oracle_pivot_df.to_csv(f'{output_dir}/crl_oracle_vs_crl_goalkde_simple_mazes_summary_table.csv')
print(f"CRL Oracle vs CRL GoalKDE summary table saved as '{output_dir}/crl_oracle_vs_crl_goalkde_simple_mazes_summary_table.csv'")

# Also save the raw combined data
oracle_combined_data = pd.concat([
    u_maze_oracle_data.assign(Environment='U-Maze'),
    big_maze_oracle_data.assign(Environment='Big Maze'),
    hardest_maze_oracle_data.assign(Environment='Hardest Maze')
], ignore_index=True)
oracle_combined_data.to_csv(f'{output_dir}/crl_oracle_vs_crl_goalkde_simple_mazes_combined_data.csv', index=False)
print(f"CRL Oracle vs CRL GoalKDE combined data saved as '{output_dir}/crl_oracle_vs_crl_goalkde_simple_mazes_combined_data.csv'")

plt.show()