import os
from pathlib import Path
import pandas as pd
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

# Set global font sizes to match create_combined_plots.py (larger fonts, same figure size)
plt.rcParams.update({
    'font.size': 32,
    'axes.titlesize': 48,
    'axes.labelsize': 42,
    'xtick.labelsize': 38,
    'ytick.labelsize': 38,
    'legend.fontsize': 38,
    'figure.titlesize': 50,
})

# Consistent method type colors (subset used here)
METHOD_TYPE_ORDER = ['GCBC', 'NN', 'FB', 'CRL', 'CIRL', 'CRL + Oracle']
METHOD_TYPE_COLOR = {t: PALETTE_COLORS[i % len(PALETTE_COLORS)] for i, t in enumerate(METHOD_TYPE_ORDER)}
# Swap colors for NN and CIRL like in combined plots
_nn_color = METHOD_TYPE_COLOR.get('NN')
_cirl_color = METHOD_TYPE_COLOR.get('CIRL')
if _nn_color is not None and _cirl_color is not None:
    METHOD_TYPE_COLOR['NN'], METHOD_TYPE_COLOR['CIRL'] = _cirl_color, _nn_color

METHOD_TYPE_ALIASES = {
    'CRL + GoalKDE (CIRL)': 'CIRL',
    'CRL + CIRL': 'CIRL',
}

def get_color_for_type(method_type: str):
    canonical = METHOD_TYPE_ALIASES.get(method_type, method_type)
    return METHOD_TYPE_COLOR.get(canonical, '#888888')


def load_method_scores(csv_path: Path) -> pd.DataFrame:
    """
    Load a CSV with columns [Method, Mean Difference, Std Error, Method Type]
    and return a DataFrame indexed by method name for easy lookup.
    """
    df = pd.read_csv(csv_path)
    df = df.set_index("Method")
    return df


def collect_points(env_to_csv: dict) -> pd.DataFrame:
    """
    For each environment, read the CSV and extract x/y points where:
      - x: imitation score of the Full Tau method
      - y: imitation score of the Mean Field method

    Methods considered (three pairs):
      - CRL + Oracle
      - CRL + GoalKDE
      - GCBC

    Returns a DataFrame with columns: [environment, method_base, x, y]
    """
    pairs = [
        ("CIRL", "CRL + GoalKDE + Full Tau", "CRL + GoalKDE + Mean Field"),
        ("GCBC", "GCBC + Full Tau", "GCBC + Mean Field"),
    ]

    rows = []
    for env_name, csv_path in env_to_csv.items():
        df = load_method_scores(csv_path)
        for method_base, full_tau_key, mean_field_key in pairs:
            if full_tau_key not in df.index:
                raise KeyError(f"Missing method '{full_tau_key}' in {csv_path}")
            if mean_field_key not in df.index:
                raise KeyError(f"Missing method '{mean_field_key}' in {csv_path}")

            # Convert from fraction (0-1) to percentage (0-100)
            x_val = float(df.loc[full_tau_key, "Mean Difference"]) * 100.0  # imitation score (Full Tau)
            y_val = float(df.loc[mean_field_key, "Mean Difference"]) * 100.0  # imitation score (Mean Field)
            rows.append({
                "environment": env_name,
                "method_base": method_base,
                "x": x_val,
                "y": y_val,
            })

    return pd.DataFrame(rows)


def plot_scatter(points: pd.DataFrame, output_dir: Path) -> None:
    """
    Create a scatter plot with:
      - x-axis: imitation score (Full Tau)
      - y-axis: imitation score (Mean Field)
    9 total points: 3 environments × 3 method bases.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Style mappings
    method_to_color = {
        "CIRL": get_color_for_type('CIRL'),
        "GCBC": get_color_for_type('GCBC'),
    }
    environments = sorted(points["environment"].unique())
    env_to_marker = {env: marker for env, marker in zip(environments, ["o", "^", "s", "D", "P"]) }

    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # Plot each point with color (method) and marker (environment)
    for _, row in points.iterrows():
        ax.scatter(
            row["x"], row["y"],
            color=method_to_color[row["method_base"]],
            marker=env_to_marker[row["environment"]],
            s=180,
            edgecolors="black",
            linewidths=1.5,
        )

    # Parity line y=x
    min_val = min(points["x"].min(), points["y"].min())
    max_val = max(points["x"].max(), points["y"].max())
    pad = 0.05 * (max_val - min_val if max_val > min_val else 1.0)
    ax.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], linestyle="--", color="gray", linewidth=3)

    # Axis labels and tick label sizes
    ax.set_xlabel("Full Tau Imitation Score (\\%)", fontsize=42)
    ax.set_ylabel("Mean Field Imitation Score (\\%)", fontsize=42)
    ax.tick_params(axis='x', labelsize=38)
    ax.tick_params(axis='y', labelsize=38)

    # Build legends: one for methods (colors), one for environments (markers)
    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([0], [0], marker="o", color="w", label=method,
               markerfacecolor=color, markeredgecolor="none", markersize=20)
        for method, color in method_to_color.items()
    ]
    env_handles = [
        Line2D([0], [0], marker=marker, color="black", label=env,
               linestyle="None", markersize=20)
        for env, marker in env_to_marker.items()
    ]

    # Analytically position legends based on measured sizes; anchor in axes coordinates
    fig = plt.gcf()

    # Create temporary legends to measure their sizes (in figure coords)
    tmp_env = ax.legend(handles=env_handles, title="Environment", loc="lower right", fontsize=38, title_fontsize=42)
    ax.add_artist(tmp_env)
    tmp_method = ax.legend(handles=method_handles, title="Method", loc="lower right", fontsize=38, title_fontsize=42)
    ax.add_artist(tmp_method)

    # Force a draw to get accurate renderer sizes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    env_bbox_fig = tmp_env.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    method_bbox_fig = tmp_method.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())

    # Remove temporary legends
    tmp_env.remove()
    tmp_method.remove()

    # Axes bounding box in figure coordinates (to convert sizes)
    ax_bbox_fig = ax.get_position()  # in figure coords

    # Convert measured legend heights (figure coords) to axes coords
    env_height_ax = env_bbox_fig.height / ax_bbox_fig.height
    # method_height_ax = method_bbox_fig.height / ax_bbox_fig.height  # not strictly needed

    # Padding in axes coordinates
    pad_x_ax = 0.008  # slightly increase gap from the axes right border
    pad_y_ax = 0.02
    inter_pad_ax = 0.01

    # Anchors in axes coordinates (loc='lower right' uses the legend's lower-right corner at the anchor)
    env_anchor_x_ax = 1.0 - pad_x_ax
    env_anchor_y_ax = 0.0 + pad_y_ax
    method_anchor_x_ax = env_anchor_x_ax
    method_anchor_y_ax = env_anchor_y_ax + env_height_ax + inter_pad_ax

    # Create final legends at computed positions using axes coordinates
    env_legend = ax.legend(
        handles=env_handles,
        title="Environment",
        loc="lower right",
        bbox_to_anchor=(env_anchor_x_ax, env_anchor_y_ax),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        fontsize=38,
        title_fontsize=42,
    )
    ax.add_artist(env_legend)

    method_legend = ax.legend(
        handles=method_handles,
        title="Method",
        loc="lower right",
        bbox_to_anchor=(method_anchor_x_ax, method_anchor_y_ax),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        fontsize=38,
        title_fontsize=42,
    )
    ax.add_artist(method_legend)

    ax.grid(True, linestyle="--", alpha=0.4, linewidth=2.0)
    plt.tight_layout()

    # Save
    out_png = output_dir / "full_tau_vs_mean_field_scatter.png"
    out_pdf = output_dir / "full_tau_vs_mean_field_scatter.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")

    # Also save the points used to plot
    points.to_csv(output_dir / "full_tau_vs_mean_field_points.csv", index=False)


def main():
    # Default CSV locations (produced by eval scripts)
    repo_root = Path(__file__).resolve().parents[1]
    env_to_csv = {
        "ant": repo_root / "notebooks" / "results_ant" / "full_trajectory_vs_mean_field_ant.csv",
        "pusher_easy": repo_root / "notebooks" / "results_pusher_easy" / "full_trajectory_vs_mean_field_pusher_easy.csv",
        "reacher": repo_root / "notebooks" / "results_reacher" / "full_trajectory_vs_mean_field_reacher.csv",
    }

    # Validate files exist
    missing = [str(p) for p in env_to_csv.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing expected CSV files: \n" + "\n".join(missing))

    points = collect_points(env_to_csv)

    # Output directory
    output_dir = repo_root / "notebooks" / "results_scatter"
    plot_scatter(points, output_dir)
    print(f"Saved scatter plot and data to: {output_dir}")


if __name__ == "__main__":
    main()


