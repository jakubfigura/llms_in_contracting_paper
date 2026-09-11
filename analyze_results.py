from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

EXPERIMENTS = {
    "Gemini 3.5 Flash": "results/experiment1",
    "Claude Haiku 4.5": "results/experiment2",
    "GPT-5.4-mini": "results/experiment3"
}

EXPERIMENTS_AFTER_PROMPT_MANIPULATION = {
    "Gemini 3.5 Flash": "results/experiment4",
    "Claude Haiku 4.5": "results/experiment5",
    "GPT-5.4-mini": "results/experiment6"
}

DATASETS = {
    "Initial prompt": (EXPERIMENTS, ""),
    "After prompt manipulation": (EXPERIMENTS_AFTER_PROMPT_MANIPULATION, "_after_prompt_manipulation"),
}

PRICE_CONSTRAINTS = (15000, 18000)

MODEL_COLORS = {
    "Gemini 3.5 Flash": "#4285F4",   # blue
    "Claude Haiku 4.5": "#D97757",   # orange
    "GPT-5.4-mini": "#6B7280",       # gray
}


def load_experiment_data(exp_path):
    """Load data from a single experiment."""
    exp_dir = Path(exp_path)
    df_results = pd.read_csv(exp_dir / "results.csv")
    df_traj = pd.read_csv(exp_dir / "price_trajectories.csv")
    df_verdicts = pd.read_csv(exp_dir / "judge_verdicts.csv")
    df_verdicts["label"] = df_verdicts["label"].astype("Int64")
    return df_results, df_traj, df_verdicts


def load_data(data_paths : dict):
    """Load data from all experiments."""
    data = {}
    for name, path in data_paths.items():
        data[name] = load_experiment_data(path)
    return data


def mean_final_round_price(df_traj):
    """Average price offered in the last round of each run.

    Buyer and seller offers coincide once they agree, so for concluded
    negotiations this is the agreed price; where a run ended without
    agreement it is the midpoint of the two standing offers.
    """
    last_round = df_traj.groupby("run")["round"].transform("max")
    last = df_traj[df_traj["round"] == last_round]
    return ((last["buyer_offer"] + last["seller_offer"]) / 2).mean()


def print_summary_single(model_name, df_results, df_traj):
    """Print statistics for a single experiment."""
    agreements = (df_results["termination_reason"] == "agreement").sum()
    print(f"=== {model_name.upper()} ===")
    print(f"Runs: {len(df_results)}")
    print(f"Agreements: {agreements} / {len(df_results)}")
    print(f"Average rounds: {df_results['rounds_completed'].mean():.2f}")
    print(f"Average price in final round: ${mean_final_round_price(df_traj):.2f}")
    print(f"Total cost: ${df_results['cost_usd'].sum():.6f}")
    print(f"Total tokens: {(df_results['input_tokens'] + df_results['output_tokens']).sum()}")
    print()


def print_summary_all(data, label):
    """Print comparative statistics for all experiments in one dataset."""
    print("\n" + "="*60)
    print(f"MODEL COMPARISON - {label.upper()}")
    print("="*60 + "\n")

    summary_list = []
    for model_name, (df_results, df_traj, _) in data.items():
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


def plot_comparative_trajectories(data, label, suffix):
    """Compare average price trajectories for all models in one dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, (_, df_traj, _) in data.items():
        traj = compute_average_trajectory(df_traj)
        buyer_means = traj[("buyer_offer", "mean")]
        seller_means = traj[("seller_offer", "mean")]
        buyer_stds = traj[("buyer_offer", "std")]
        seller_stds = traj[("seller_offer", "std")]

        rounds = buyer_means.index
        ax.plot(rounds, buyer_means, marker="o", linestyle="-", color=MODEL_COLORS[model_name],
                linewidth=2, label=f"{model_name} (buyer)")
        ax.fill_between(rounds, buyer_means - buyer_stds, buyer_means + buyer_stds,
                        color=MODEL_COLORS[model_name], alpha=0.15)

        ax.plot(rounds, seller_means, marker="s", linestyle="--", color=MODEL_COLORS[model_name],
                linewidth=2, label=f"{model_name} (seller)")
        ax.fill_between(rounds, seller_means - seller_stds, seller_means + seller_stds,
                        color=MODEL_COLORS[model_name], alpha=0.15)

    for i, constraint in enumerate(PRICE_CONSTRAINTS):
        ax.axhline(constraint, color="black", linestyle="--", linewidth=1.5, alpha=0.8,
                   label="price constraints" if i == 0 else None)

    ax.set_xlabel("Negotiation Round", fontsize=11)
    ax.set_ylabel("Mean Offered Price", fontsize=11)
    ax.set_title(f"Comparison of Mean Price Trajectories Between Models ({label})", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    outfile = f"results/comparison_price_trajectories{suffix}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


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


def filter_final_round_verdicts(df_verdicts):
    """Keep only the verdicts from the last round of each run."""
    last_round = df_verdicts.groupby("run")["round"].transform("max")
    return df_verdicts[df_verdicts["round"] == last_round]


def plot_comparative_criteria_satisfaction(data, label, suffix, final_round_only=False):
    """Create separate plots for each model showing criteria satisfaction."""
    if final_round_only:
        title = f"Contract Criteria Evaluation in Final Rounds by Model ({label})"
        ylabel = "Number of Runs"
        outfile = f"results/comparison_criteria_satisfaction_final_round{suffix}.png"
    else:
        title = f"Legal Criteria Satisfaction by Model, Absolute Numbers ({label})"
        ylabel = "Number of Evaluations"
        outfile = f"results/comparison_criteria_satisfaction{suffix}.png"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    models = list(data.keys())

    status_colors = {
        "satisfied": "#2ecc71",      # green
        "not_satisfied": "#e74c3c",  # red
        "unable": "#95a5a6"          # gray
    }

    criteria_per_model = {}
    all_criteria = set()

    for model_name, (_, _, df_verdicts) in data.items():
        if final_round_only:
            df_verdicts = filter_final_round_verdicts(df_verdicts)
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
        ax.set_title(model_name, fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"C{i}" for i in all_criteria], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=10)

    fig.suptitle(title, fontsize=13, y=0.98)

    # Add shared legend below title
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=status_colors["satisfied"], alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, fc=status_colors["not_satisfied"], alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, fc=status_colors["unable"], alpha=0.85)
    ]
    fig.legend(handles, ["Satisfied", "Not Satisfied", "Unable to Determine"],
               loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.92))

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return outfile


def print_criteria_comparison(data, label):
    """Print comparative criteria statistics for one dataset."""
    print("\n" + "="*80)
    print(f"CRITERIA SATISFACTION COMPARISON - {label.upper()} (Including Unable to Determine)")
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


def label_distribution(df_verdicts):
    """Share of satisfied / violated / undetermined labels over all verdicts."""
    labels = df_verdicts["label"]
    n = len(labels)
    return {
        "satisfied": (labels == 1).fillna(False).sum() / n,
        "violated": (labels == 0).fillna(False).sum() / n,
        "undetermined": labels.isna().sum() / n,
    }


def label_counts(df_verdicts):
    """Counts of satisfied / not satisfied / unable labels pooled over all criteria."""
    labels = df_verdicts["label"]
    return {
        "total": len(labels),
        "satisfied": int((labels == 1).fillna(False).sum()),
        "not_satisfied": int((labels == 0).fillna(False).sum()),
        "unable": int(labels.isna().sum()),
    }


def print_overall_label_distribution(data, label, final_round_only=False):
    """Share of each verdict type per model, pooled over all criteria.

    Covers every round of every run, or only the last round of each run when
    ``final_round_only`` is set.
    """
    scope = "FINAL ROUND ONLY" if final_round_only else "ALL ROUNDS"
    print("\n" + "="*80)
    print(f"OVERALL VERDICT DISTRIBUTION PER MODEL - {label.upper()} ({scope})")
    print("="*80 + "\n")

    rows = []
    for model_name, (_, _, df_verdicts) in data.items():
        if final_round_only:
            df_verdicts = filter_final_round_verdicts(df_verdicts)
        counts = label_counts(df_verdicts)
        total = counts["total"]
        rows.append({
            "Model": model_name,
            "Total Evaluations": total,
            "Satisfied": counts["satisfied"],
            "Satisfied %": f"{100*counts['satisfied']/total:.1f}%",
            "Not Satisfied": counts["not_satisfied"],
            "Not Satisfied %": f"{100*counts['not_satisfied']/total:.1f}%",
            "Unable": counts["unable"],
            "Unable %": f"{100*counts['unable']/total:.1f}%",
        })

    print(pd.DataFrame(rows).to_string(index=False))
    print()


def analyze_dataset(label, experiments, suffix):
    """Run the full analysis for a single dataset and return its loaded data."""
    data = load_data(experiments)

    print("\n" + "="*80)
    print(f"COMPARATIVE ANALYSIS OF EXPERIMENTS - {label.upper()}")
    print("="*80)

    for model_name, (df_results, df_traj, _) in data.items():
        print_summary_single(model_name, df_results, df_traj)

    print_summary_all(data, label)
    print_overall_label_distribution(data, label)
    print_overall_label_distribution(data, label, final_round_only=True)
    print_criteria_comparison(data, label)

    print(f"\nCreating comparative plots for {label.lower()}...")
    outfiles = [
        plot_comparative_trajectories(data, label, suffix),
        plot_comparative_criteria_satisfaction(data, label, suffix),
        plot_comparative_criteria_satisfaction(data, label, suffix, final_round_only=True),
    ]
    for outfile in outfiles:
        print(f"  {outfile}")

    return data


def print_dataset_comparison(datasets):
    """Compare the same models across datasets, dataset by dataset."""
    labels = list(datasets.keys())
    if len(labels) < 2:
        return

    print("\n" + "="*80)
    print("CROSS-DATASET COMPARISON")
    print("="*80)

    models = list(datasets[labels[0]].keys())

    rows = []
    for model_name in models:
        for label in labels:
            df_results, df_traj, df_verdicts = datasets[label][model_name]
            agreements = (df_results["termination_reason"] == "agreement").sum()
            dist = label_distribution(df_verdicts)
            rows.append({
                "Model": model_name,
                "Dataset": label,
                "Runs": len(df_results),
                "Agreement Rate": f"{100*agreements/len(df_results):.1f}%",
                "Avg Rounds": f"{df_results['rounds_completed'].mean():.2f}",
                "Avg Final Round Price": f"${mean_final_round_price(df_traj):.2f}",
                "Satisfied": f"{dist['satisfied']:.1%}",
                "Violated": f"{dist['violated']:.1%}",
                "Undetermined": f"{dist['undetermined']:.1%}",
            })

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print()


def main():
    datasets = {}
    for label, (experiments, suffix) in DATASETS.items():
        datasets[label] = analyze_dataset(label, experiments, suffix)

    print_dataset_comparison(datasets)


if __name__ == "__main__":
    main()
