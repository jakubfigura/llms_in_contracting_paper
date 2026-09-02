from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENTS = {
    "Gemini 3.5 Flash": "results/experiment1",
    "Claude Haiku 4.5": "results/experiment2",
    "GPT-5.4-mini": "results/experiment3"
}


def load_experiment_data(exp_path):
    """Load data from a single experiment."""
    exp_dir = Path(exp_path)
    df_results = pd.read_csv(exp_dir / "results.csv")
    df_traj = pd.read_csv(exp_dir / "price_trajectories.csv")
    df_verdicts = pd.read_csv(exp_dir / "judge_verdicts.csv")
    df_verdicts["label"] = df_verdicts["label"].astype("Int64")
    return df_results, df_traj, df_verdicts


def load_all_data():
    """Load data from all experiments."""
    data = {}
    for name, path in EXPERIMENTS.items():
        data[name] = load_experiment_data(path)
    return data


def print_summary_single(model_name, df_results):
    """Print statistics for a single experiment."""
    agreements = (df_results["termination_reason"] == "agreement").sum()
    print(f"=== {model_name.upper()} ===")
    print(f"Runs: {len(df_results)}")
    print(f"Agreements: {agreements} / {len(df_results)}")
    print(f"Average rounds: {df_results['rounds_completed'].mean():.2f}")
    print(f"Total cost: ${df_results['cost_usd'].sum():.6f}")
    print(f"Total tokens: {(df_results['input_tokens'] + df_results['output_tokens']).sum()}")
    print()


def print_summary_all(data):
    """Print comparative statistics for all experiments."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60 + "\n")

    summary_list = []
    for model_name, (df_results, _, _) in data.items():
        agreements = (df_results["termination_reason"] == "agreement").sum()
        summary_list.append({
            "Model": model_name,
            "Runs": len(df_results),
            "Agreements": agreements,
            "Agreement Rate": f"{100*agreements/len(df_results):.1f}%",
            "Avg Rounds": f"{df_results['rounds_completed'].mean():.2f}",
            "Avg Final Price": f"${df_results['final_price'].mean():.2f}" if (df_results['final_price'] > 0).any() else "N/A",
            "Total Cost": f"${df_results['cost_usd'].sum():.4f}",
        })

    summary_df = pd.DataFrame(summary_list)
    print(summary_df.to_string(index=False))
    print()


def compute_average_trajectory(df_traj):
    """Compute average price trajectory across all runs."""
    traj_by_round = df_traj.groupby("round").agg({
        "buyer_offer": ["mean", "std"],
        "seller_offer": ["mean", "std"]
    })
    return traj_by_round


def plot_comparative_trajectories(data):
    """Compare average price trajectories for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Gemini 3.5 Flash": "#4285F4", "Claude Haiku 4.5": "#000000", "GPT-5.4-mini": "#00A67E"}

    for model_name, (_, df_traj, _) in data.items():
        traj = compute_average_trajectory(df_traj)
        buyer_means = traj[("buyer_offer", "mean")]
        seller_means = traj[("seller_offer", "mean")]
        buyer_stds = traj[("buyer_offer", "std")]
        seller_stds = traj[("seller_offer", "std")]

        rounds = buyer_means.index
        ax.plot(rounds, buyer_means, marker="o", linestyle="-", color=colors[model_name],
                linewidth=2, label=f"{model_name} (buyer)")
        ax.fill_between(rounds, buyer_means - buyer_stds, buyer_means + buyer_stds,
                        color=colors[model_name], alpha=0.15)

        ax.plot(rounds, seller_means, marker="s", linestyle="--", color=colors[model_name],
                linewidth=2, label=f"{model_name} (seller)")
        ax.fill_between(rounds, seller_means - seller_stds, seller_means + seller_stds,
                        color=colors[model_name], alpha=0.15)

    ax.set_xlabel("Negotiation Round", fontsize=11)
    ax.set_ylabel("Average Offered Price", fontsize=11)
    ax.set_title("Comparison of Average Price Trajectories Between Models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/comparison_price_trajectories.png", dpi=150)
    plt.close(fig)


def print_criteria_stats_single(model_name, df_verdicts):
    """Compute criteria statistics for a single experiment."""
    criteria_stats = (
        df_verdicts
        .groupby("criterion")
        .agg(
            total=("label", "size"),
            satisfied=("label", lambda s: (s == 1).sum()),
            not_satisfied=("label", lambda s: (s == 0).sum()),
            unable_to_determine=("label", lambda s: s.isna().sum()),
        )
    )
    # Add proportions
    criteria_stats["satisfied_pct"] = criteria_stats["satisfied"] / criteria_stats["total"]
    criteria_stats["not_satisfied_pct"] = criteria_stats["not_satisfied"] / criteria_stats["total"]
    criteria_stats["unable_pct"] = criteria_stats["unable_to_determine"] / criteria_stats["total"]
    return criteria_stats


def plot_comparative_criteria_satisfaction(data):
    """Create separate plots for each model showing criteria satisfaction."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    models = list(data.keys())
    colors = {"Gemini 3.5 Flash": "#4285F4", "Claude Haiku 4.5": "#000000", "GPT-5.4-mini": "#00A67E"}

    status_colors = {
        "satisfied": "#2ecc71",      # green
        "not_satisfied": "#e74c3c",  # red
        "unable": "#95a5a6"          # gray
    }

    criteria_per_model = {}
    all_criteria = set()

    for model_name, (_, _, df_verdicts) in data.items():
        stats = print_criteria_stats_single(model_name, df_verdicts)
        criteria_per_model[model_name] = stats
        all_criteria.update(stats.index)

    all_criteria = sorted(all_criteria)

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        stats = criteria_per_model[model_name]

        x_pos = np.arange(len(all_criteria))
        width = 0.6

        satisfied = [stats.loc[c, "satisfied"] for c in all_criteria]
        not_satisfied = [stats.loc[c, "not_satisfied"] for c in all_criteria]
        unable = [stats.loc[c, "unable_to_determine"] for c in all_criteria]

        ax.bar(x_pos, satisfied, width, label="Satisfied",
               color=status_colors["satisfied"], alpha=0.85)
        ax.bar(x_pos, not_satisfied, width, bottom=satisfied,
               label="Not Satisfied", color=status_colors["not_satisfied"], alpha=0.85)
        ax.bar(x_pos, unable, width, bottom=np.array(satisfied) + np.array(not_satisfied),
               label="Unable to Determine", color=status_colors["unable"], alpha=0.85)

        ax.set_xlabel("Criterion", fontsize=10)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"C{i}" for i in all_criteria], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Number of Evaluations", fontsize=10)

    fig.suptitle("Legal Criteria Satisfaction by Model (Absolute Numbers)",
                 fontsize=13, fontweight="bold", y=0.98)

    # Add shared legend below title
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=status_colors["satisfied"], alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, fc=status_colors["not_satisfied"], alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, fc=status_colors["unable"], alpha=0.85)
    ]
    fig.legend(handles, ["Satisfied", "Not Satisfied", "Unable to Determine"],
               loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.92))

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig("results/comparison_criteria_satisfaction.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_criteria_comparison(data):
    """Print comparative criteria statistics."""
    print("\n" + "="*80)
    print("CRITERIA SATISFACTION COMPARISON (Including Unable to Determine)")
    print("="*80 + "\n")

    all_criteria = set()
    criteria_per_model = {}

    for model_name, (_, _, df_verdicts) in data.items():
        stats = print_criteria_stats_single(model_name, df_verdicts)
        criteria_per_model[model_name] = stats
        all_criteria.update(stats.index)

    for criterion in sorted(all_criteria):
        print(f"\n--- Criterion {criterion} ---")
        for model_name in data.keys():
            if criterion in criteria_per_model[model_name].index:
                sat = criteria_per_model[model_name].loc[criterion, "satisfied"]
                not_sat = criteria_per_model[model_name].loc[criterion, "not_satisfied"]
                unable = criteria_per_model[model_name].loc[criterion, "unable_to_determine"]
                total = criteria_per_model[model_name].loc[criterion, "total"]
                sat_pct = criteria_per_model[model_name].loc[criterion, "satisfied_pct"]
                print(f"  {model_name:20s}: Satisfied={sat:3d} ({sat_pct:5.1%}) | Not Satisfied={not_sat:3d} | Unable={unable:3d} (total={total})")


def main():
    data = load_all_data()

    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS OF EXPERIMENTS")
    print("="*80)

    for model_name, (df_results, _, _) in data.items():
        print_summary_single(model_name, df_results)

    print_summary_all(data)
    print_criteria_comparison(data)

    print("\n" + "="*80)
    print("Creating comparative plots...")
    print("="*80)

    plot_comparative_trajectories(data)
    plot_comparative_criteria_satisfaction(data)

    print("\nComparative plots saved to:")
    print("  results/comparison_price_trajectories.png")
    print("  results/comparison_criteria_satisfaction.png")


if __name__ == "__main__":
    main()
