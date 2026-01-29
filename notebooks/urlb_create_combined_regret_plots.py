import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(['science', 'ieee'])
try:
    from palettable.colorbrewer.qualitative import Set2_7  # noqa: F401
    PALETTE_COLORS = Set2_7.mpl_colors
except Exception:
    PALETTE_COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']

# Global font sizes to match create_combined_plots
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

# Consistent color mapping by method type
METHOD_TYPE_ORDER = ['GCBC', 'NN', 'FB', 'CRL', 'CIRL', 'CRL + Oracle', 'HILP']
METHOD_TYPE_COLOR = {t: PALETTE_COLORS[i % len(PALETTE_COLORS)] for i, t in enumerate(METHOD_TYPE_ORDER)}
# Swap colors for NN and CIRL (same as create_combined_plots)
_nn_color = METHOD_TYPE_COLOR.get('NN')
_cirl_color = METHOD_TYPE_COLOR.get('CIRL')
if _nn_color is not None and _cirl_color is not None:
    METHOD_TYPE_COLOR['NN'], METHOD_TYPE_COLOR['CIRL'] = _cirl_color, _nn_color

# Match CRL + Oracle color to the tan used elsewhere
try:
    METHOD_TYPE_COLOR['CRL + Oracle'] = PALETTE_COLORS[6]
except Exception:
    METHOD_TYPE_COLOR['CRL + Oracle'] = '#e5c494'
# Ensure HILP has a distinct color from CRL + Oracle
METHOD_TYPE_COLOR['HILP'] = '#9467bd'

METHOD_TYPE_ALIASES = {
    'CRL + GoalKDE (CIRL)': 'CIRL',
    'CRL + CIRL': 'CIRL',
}

def get_color_for_type(method_type: str):
    canonical = METHOD_TYPE_ALIASES.get(method_type, method_type)
    return METHOD_TYPE_COLOR.get(canonical, '#888888')


def find_regret_csvs(base_dir: str):
    patterns = [
        # New extended CSVs (preferred)
        os.path.join(base_dir, 'results_*', 'antforward_regret_goalkde_vs_fb_hilp_psm.csv'),
        os.path.join(base_dir, 'results_*', 'antjump_regret_goalkde_vs_fb_hilp_psm.csv'),
        os.path.join(base_dir, 'results_*', 'antflip_regret_goalkde_vs_fb_hilp_psm.csv'),
        # Legacy CSVs (fallback)
        os.path.join(base_dir, 'results_*', 'antforward_regret_goalkde_vs_fb.csv'),
        os.path.join(base_dir, 'results_*', 'antjump_regret_goalkde_vs_fb.csv'),
        os.path.join(base_dir, 'results_*', 'antflip_regret_goalkde_vs_fb.csv'),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return files


def load_regret(csv_path: str):
    df = pd.read_csv(csv_path)
    # Expect columns: Method, MeanRegret, StdError
    # Normalize method names
    df['Method'] = df['Method'].str.strip()
    # Convert to float
    df['MeanRegret'] = pd.to_numeric(df['MeanRegret'], errors='coerce')
    df['StdError'] = pd.to_numeric(df['StdError'], errors='coerce')
    return df


def infer_expert_from_path(path: str) -> str:
    fname = os.path.basename(path)
    if fname.startswith('antforward_'):
        return 'antforward'
    if fname.startswith('antjump_'):
        return 'antjump'
    if fname.startswith('antflip_'):
        return 'antflip'
    # fallback: parse parent data
    parts = fname.split('_')
    for p in parts:
        if p in ('antforward', 'antjump'):
            return p
    return 'unknown'


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'combined_results')
    os.makedirs(out_dir, exist_ok=True)

    csvs = find_regret_csvs(base_dir)
    if not csvs:
        print('No regret CSVs found under notebooks/results_*/')
        return

    # Group latest per expert policy (if multiple inference methods exist, pick most recent file)
    by_expert = {}
    for f in csvs:
        expert = infer_expert_from_path(f)
        mtime = os.path.getmtime(f)
        if expert not in by_expert or mtime > by_expert[expert]['mtime']:
            by_expert[expert] = {'path': f, 'mtime': mtime}

    expert_to_title = {
        'antforward': 'Ant Forward',
        'antjump': 'Ant Jump',
        'antflip': 'Ant Flip',
    }

    # Desired order: FB, then HILP, then MainMF (CRL + Oracle), CRL + GoalKDE (CIRL)
    methods_order = ['FB', 'HILP', 'MainMF', 'CRL + GoalKDE']
    display_label = {
        'FB': 'FB',
        'MainMF': 'CRL + Oracle',
        'CRL + GoalKDE': 'CRL + GoalKDE (CIRL)',
        'HILP': 'HILP',
    }
    # Colors by method TYPE to match create_combined_plots
    type_for_method = {
        'FB': 'FB',
        'MainMF': 'CRL + Oracle',
        'CRL + GoalKDE': 'CIRL',
        'HILP': 'HILP',
    }

    # Build a single combined dataframe
    combined_rows = []
    environments = []
    for expert in ['antforward', 'antjump', 'antflip']:
        if expert in by_expert:
            df = load_regret(by_expert[expert]['path'])
            df = df[df['Method'].isin(methods_order)].copy()
            if df.empty:
                continue
            env_name = expert_to_title.get(expert, expert)
            environments.append(env_name)
            for _, r in df.iterrows():
                combined_rows.append({
                    'Environment': env_name,
                    'Method': r['Method'],
                    'MeanRegret': r['MeanRegret'],
                    'StdError': r['StdError'],
                    'Method Type': type_for_method.get(r['Method'], r['Method'])
                })

    if not combined_rows:
        print('No valid data to plot for URLB regret combined plot.')
        return

    combined_df = pd.DataFrame(combined_rows)
    # Order environments consistently
    environments = [expert_to_title[e] for e in ['antforward', 'antjump', 'antflip'] if expert_to_title[e] in environments]

    # Prepare 3 subplots (one per environment)
    num_subplots = 3
    fig, axes = plt.subplots(1, num_subplots, figsize=(24, 8), squeeze=False)
    axes = axes[0]

    # Methods present and sort by methods_order
    all_methods = combined_df['Method'].unique().tolist()
    def _method_sort_key(m):
        return (methods_order.index(m) if m in methods_order else len(methods_order), str(m))
    methods = sorted(all_methods, key=_method_sort_key)

    # Color palette by method type
    method_palette = {m: get_color_for_type(type_for_method[m]) for m in methods}

    # Per-subplot grouped bars and per-env y limits
    group_width = 0.8
    per_env_limits = []
    visible_env_names = []
    visible_axes = []
    added_ylabel = False
    for i in range(num_subplots):
        ax_i = axes[i]
        if i < len(environments):
            env = environments[i]
            env_df = combined_df[combined_df['Environment'] == env]
            env_methods = [m for m in methods if m in env_df['Method'].values]
            if env_methods:
                num_methods = len(env_methods)
                bar_width = group_width / max(1, num_methods)
                base_x = 0.0
                offsets = (np.arange(num_methods) - (num_methods - 1) / 2.0) * bar_width
                for offset, m in zip(offsets, env_methods):
                    row = env_df[env_df['Method'] == m].iloc[0]
                    height = row['MeanRegret']
                    err = row['StdError']
                    xpos = base_x + offset
                    ax_i.bar(xpos, height, width=bar_width * 0.9, color=method_palette[m], edgecolor='none', linewidth=0)
                    ax_i.errorbar(xpos, height, yerr=err, fmt='none', ecolor='black', capsize=5, linewidth=1.0)
                # X-axis: single tick with environment name
                ax_i.set_xticks([base_x])
                ax_i.set_xticklabels([env], fontsize=38, rotation=0)
            else:
                ax_i.set_xticks([0.0])
                ax_i.set_xticklabels([env], fontsize=38, rotation=0)

            # Y-axis formatting and grid per subplot
            from matplotlib.ticker import AutoMinorLocator
            ax_i.yaxis.set_minor_locator(AutoMinorLocator(5))
            if not added_ylabel:
                ax_i.set_ylabel('Regret', fontsize=42)
                added_ylabel = True
            ax_i.tick_params(axis='y', labelsize=38)
            ax_i.grid(axis='y', which='major', linestyle='--', alpha=0.7)
            ax_i.grid(axis='y', which='minor', linestyle=':', alpha=0.3)

            # Per-env Y limits based on that env's data, include zero and small padding
            if not env_df.empty:
                min_val_i = float(np.nanmin(env_df['MeanRegret'] - env_df['StdError']))
                max_val_i = float(np.nanmax(env_df['MeanRegret'] + env_df['StdError']))
                pad_i = 0.05 * max(1.0, max(abs(min_val_i), max_val_i))
                ymin_i = min(0.0, min_val_i - pad_i)
                ymax_i = max_val_i + pad_i
                ax_i.set_ylim(ymin_i, ymax_i)
                per_env_limits.append((ymin_i, ymax_i))
                visible_axes.append(ax_i)
                visible_env_names.append(env)
        else:
            ax_i.set_visible(False)

    # Align y=0 across visible subplots while keeping each subplot's span
    if per_env_limits and visible_axes:
        spans = [ymax - ymin for (ymin, ymax) in per_env_limits]
        # Compute current zero positions as fraction of span; choose common target (median) to minimize shifts
        zero_fracs = [(0.0 - ymin) / span if span != 0 else 0.5 for (ymin, ymax), span in zip(per_env_limits, spans)]
        from statistics import median
        target_frac = float(median(zero_fracs))
        # Clamp target within [0.05, 0.95] to keep some headroom
        target_frac = max(0.05, min(0.95, target_frac))
        # Apply adjusted limits preserving each span
        for ax_i, span, env_name in zip(visible_axes, spans, visible_env_names):
            # Start with aligned limits
            new_span = span
            # Custom tick sets per environment (fixed for Ant Forward/Ant Jump)
            if env_name == 'Ant Forward':
                custom_ticks = [0, 300, 600, 900]
            elif env_name == 'Ant Jump':
                custom_ticks = [0, 40, 80, 120]
            elif env_name == 'Ant Flip':
                custom_ticks = [0, 200, 400, 600]
            else:
                custom_ticks = []  # fallback to automatic if unknown

            # Ensure the aligned limits include all requested ticks while keeping zero at the same relative position
            if custom_ticks:
                min_tick = float(min(custom_ticks))
                max_tick = float(max(custom_ticks))
                # Lower bounds on span needed to include ticks with zero at target_frac
                if target_frac > 0:
                    lb_min = max(0.0, -min_tick / target_frac)
                else:
                    lb_min = 0.0
                if (1.0 - target_frac) > 0:
                    lb_max = max_tick / (1.0 - target_frac)
                else:
                    lb_max = new_span
                new_span = max(new_span, lb_min, lb_max)
            new_ymin = 0.0 - target_frac * new_span
            new_ymax = new_ymin + new_span
            ax_i.set_ylim(new_ymin, new_ymax)

            # Apply fixed custom ticks if provided
            from matplotlib.ticker import FixedLocator, FormatStrFormatter
            if custom_ticks:
                ax_i.yaxis.set_major_locator(FixedLocator(custom_ticks))
                ax_i.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # Legend by Method Type
    from matplotlib.patches import Patch
    type_to_color = {}
    for m in methods:
        t = type_for_method.get(m, m)
        if t not in type_to_color:
            type_to_color[t] = get_color_for_type(t)
    type_order = ['FB', 'HILP', 'CRL + Oracle', 'CIRL']
    ordered_types = [t for t in type_order if t in type_to_color]
    legend_handles = []
    for t in ordered_types:
        label = t
        if t == 'CIRL':
            label = 'CRL + GoalKDE (CIRL)'
        legend_handles.append(Patch(facecolor=type_to_color[t], edgecolor='none', label=label))
    if legend_handles:
        # Common legend at the bottom
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(legend_handles),
            fontsize=38,
            title=None,
            frameon=False
        )

    plt.tight_layout()
    # Add extra bottom margin to accommodate bottom legend
    fig.subplots_adjust(bottom=0.2)
    out_png = os.path.join(out_dir, 'urlb_regret_combined.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved combined regret plot to: {out_png}')


if __name__ == '__main__':
    main()


