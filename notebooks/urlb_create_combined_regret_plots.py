import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
METHOD_TYPE_ORDER = ['GCBC', 'NN', 'FB', 'CRL', 'CIRL', 'CRL + Oracle']
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

METHOD_TYPE_ALIASES = {
    'CRL + GoalKDE (CIRL)': 'CIRL',
    'CRL + CIRL': 'CIRL',
}

def get_color_for_type(method_type: str):
    canonical = METHOD_TYPE_ALIASES.get(method_type, method_type)
    return METHOD_TYPE_COLOR.get(canonical, '#888888')


def find_regret_csvs(base_dir: str):
    patterns = [
        os.path.join(base_dir, 'results_*', 'antforward_regret_goalkde_vs_fb.csv'),
        os.path.join(base_dir, 'results_*', 'antjump_regret_goalkde_vs_fb.csv'),
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
    }

    # Desired order: FB, MainMF (CRL + Oracle), CRL + GoalKDE (CIRL)
    methods_order = ['FB', 'MainMF', 'CRL + GoalKDE']
    display_label = {
        'FB': 'FB',
        'MainMF': 'CRL + Oracle',
        'CRL + GoalKDE': 'CRL + GoalKDE (CIRL)'
    }
    # Colors by method TYPE to match create_combined_plots
    type_for_method = {
        'FB': 'FB',
        'MainMF': 'CRL + Oracle',
        'CRL + GoalKDE': 'CIRL'
    }

    # Build a single combined dataframe
    combined_rows = []
    environments = []
    for expert in ['antforward', 'antjump']:
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
    environments = [expert_to_title[e] for e in ['antforward', 'antjump'] if expert_to_title[e] in environments]

    # Prepare single-axis combined plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Methods present and sort by methods_order
    all_methods = combined_df['Method'].unique().tolist()
    def _method_sort_key(m):
        return (methods_order.index(m) if m in methods_order else len(methods_order), str(m))
    methods = sorted(all_methods, key=_method_sort_key)

    # Color palette by method type
    method_palette = {m: get_color_for_type(type_for_method[m]) for m in methods}

    # Compute positions
    env_positions = np.arange(len(environments))
    max_methods_per_env = max((combined_df[combined_df['Environment'] == env]['Method'].nunique() for env in environments))
    group_width = 0.8
    bar_width = group_width / max_methods_per_env if max_methods_per_env > 0 else 0.4

    # Plot grouped bars
    for i, env in enumerate(environments):
        env_df = combined_df[combined_df['Environment'] == env]
        env_methods = [m for m in methods if m in env_df['Method'].values]
        if not env_methods:
            continue
        offsets = (np.arange(len(env_methods)) - (len(env_methods) - 1) / 2.0) * bar_width
        for offset, m in zip(offsets, env_methods):
            row = env_df[env_df['Method'] == m].iloc[0]
            height = row['MeanRegret']
            err = row['StdError']
            xpos = env_positions[i] + offset
            ax.bar(xpos, height, width=bar_width * 0.9, color=method_palette[m], edgecolor='none', linewidth=0)
            ax.errorbar(xpos, height, yerr=err, fmt='none', ecolor='black', capsize=5, linewidth=1.0)

    # X-axis formatting with environment labels
    ax.set_xticks(env_positions)
    ax.set_xticklabels(environments, fontsize=38, rotation=0)
    ax.set_xlabel('')

    # Y-axis formatting with minor ticks
    from matplotlib.ticker import AutoMinorLocator
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_ylabel('Regret', fontsize=42)
    ax.tick_params(axis='y', labelsize=38)
    ax.grid(axis='y', which='major', linestyle='--', alpha=0.7)
    ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3)

    # Y limits with headroom and room for possible negatives due to error bars
    min_val = float(np.nanmin(combined_df['MeanRegret'] - combined_df['StdError']))
    max_val = float(np.nanmax(combined_df['MeanRegret'] + combined_df['StdError']))
    pad = 0.05 * max(1.0, max(abs(min_val), max_val))
    ymin = min(0.0, min_val - pad)
    ymax = max_val + pad
    # Round to neat tens
    ymin = float(int(np.floor(ymin / 10.0)) * 10)
    ymax = float(int(np.ceil(ymax / 10.0)) * 10)
    ax.set_ylim(ymin, ymax)

    # Legend by Method Type
    from matplotlib.patches import Patch
    type_to_color = {}
    for m in methods:
        t = type_for_method.get(m, m)
        if t not in type_to_color:
            type_to_color[t] = get_color_for_type(t)
    type_order = ['FB', 'CRL + Oracle', 'CIRL']
    ordered_types = [t for t in type_order if t in type_to_color]
    legend_handles = [Patch(facecolor=type_to_color[t], edgecolor='none', label=t if t != 'CIRL' else 'CRL + GoalKDE (CIRL)') for t in ordered_types]
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
            frameon=False
        )

    plt.tight_layout()
    out_png = os.path.join(out_dir, 'urlb_regret_combined.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved combined regret plot to: {out_png}')


if __name__ == '__main__':
    main()


