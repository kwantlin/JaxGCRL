import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    # Prepare figure with two subplots: Ant Forward, Ant Jump
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    expert_to_title = {
        'antforward': 'Ant Forward',
        'antjump': 'Ant Jump',
    }

    # Desired order: FB, MainMF (CRL + Oracle), CRL + GoalKDE
    methods_order = ['FB', 'MainMF', 'CRL + GoalKDE']
    color_map = {
        'FB': '#2ca02c',            # green
        'MainMF': '#1f77b4',        # blue
        'CRL + GoalKDE': '#ff7f0e'  # orange
    }

    for i, expert in enumerate(['antforward', 'antjump']):
        ax = axes[i]
        if expert in by_expert:
            df = load_regret(by_expert[expert]['path'])
            # Align to expected order and filter present methods
            df = df[df['Method'].isin(methods_order)].copy()
            df['Method'] = pd.Categorical(df['Method'], categories=methods_order, ordered=True)
            df = df.sort_values('Method')
            means = df['MeanRegret'].to_numpy()
            errs = df['StdError'].to_numpy()
            x = np.arange(len(df))
            bar_colors = [color_map[m] for m in df['Method']]
            ax.bar(x, means, yerr=errs, capsize=6, color=bar_colors)
            ax.set_xticks(x)
            # Remove x-axis labels per request
            ax.set_xlabel('')
            ax.set_xticklabels([])
            # Add legend only to the first subplot
            if i == 0:
                from matplotlib.patches import Patch
                legend_handles = [
                    Patch(facecolor=color_map['FB'], label='FB'),
                    Patch(facecolor=color_map['MainMF'], label='CRL + Oracle'),
                    Patch(facecolor=color_map['CRL + GoalKDE'], label='CRL + GoalKDE'),
                ]
                ax.legend(handles=legend_handles, loc='lower left')
        ax.set_title(expert_to_title.get(expert, expert), fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylabel('Regret') if i == 0 else None

    plt.tight_layout()
    out_png = os.path.join(out_dir, 'urlb_regret_combined.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved combined regret plot to: {out_png}')


if __name__ == '__main__':
    main()


