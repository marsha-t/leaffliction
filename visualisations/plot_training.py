
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


MODELS = {
    "custom_cnn": "Custom CNN",
    "pretrained_cnn": "ResNet18",
    "custom_vit": "Custom ViT",
    "pretrained_vit": "ViT-B/16",
}

DATASETS = ("apple", "grape")

COLORS = {
    "custom_cnn": "tab:blue",
    "pretrained_cnn": "tab:orange",
    "custom_vit": "tab:green",
    "pretrained_vit": "tab:red",
}

OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parent


def load_histories(directory):
    """Load training histories from checkpoint directories."""
    histories = {}

    for dataset in DATASETS:
        histories[dataset] = {}

        for model in MODELS:
            path = (
                directory
                / model
                / f"baseline_{dataset}"
                / "history.json"
            )

            with path.open(encoding="utf-8") as file:
                history = json.load(file)

            keys = (
                "train_loss",
                "validation_loss",
                "train_accuracy",
                "validation_accuracy",
            )

            lengths = [len(history[key]) for key in keys]

            if not lengths[0] or len(set(lengths)) != 1:
                raise ValueError(
                    f"Inconsistent or empty history: {path}"
                )

            histories[dataset][model] = history

    return histories


def best_epoch_index(history):
    """Return the index of the minimum validation loss."""
    return min(
        range(len(history["validation_loss"])),
        key=lambda i: history["validation_loss"][i],
    )


def plot_metric(histories, metric, validation_only=False):
    """Plot training histories for Apple and Grape."""
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4.5),
        sharey=True,
        constrained_layout=True,
    )

    for ax, dataset in zip(axes, DATASETS):
        for model in MODELS:
            history = histories[dataset][model]
            epochs = range(1, len(history["train_loss"]) + 1)
            color = COLORS[model]

            train_values = history[f"train_{metric}"]
            val_values = history[f"validation_{metric}"]

            if metric == "accuracy":
                train_values = [v * 100 for v in train_values]
                val_values = [v * 100 for v in val_values]

            if not validation_only:
                ax.plot(
                    epochs,
                    train_values,
                    color=color,
                    linestyle="-",
                    linewidth=1.7,
                    alpha=0.8,
                )

            ax.plot(
                epochs,
                val_values,
                color=color,
                linestyle="-"
                if validation_only else "--",
                linewidth=2,
            )

            best_index = best_epoch_index(history)

            ax.scatter(
                best_index + 1,
                val_values[best_index],
                color=color,
                s=28,
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )

        ax.set_title(dataset.capitalize())
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=5)
        )
        ax.grid(alpha=0.15)

    axes[0].set_ylabel(
        "Accuracy (%)" if metric == "accuracy"
        else "Cross-entropy loss"
    )

    model_handles = [
        Line2D(
            [0], [0],
            color=COLORS[model],
            lw=2,
            label=label,
        )
        for model, label in MODELS.items()
    ]

    legend_handles = model_handles

    if not validation_only:
        legend_handles += [
            Line2D(
                [0], [0],
                color="gray",
                lw=2,
                linestyle="-",
                label="Training",
            ),
            Line2D(
                [0], [0],
                color="gray",
                lw=2,
                linestyle="--",
                label="Validation",
            ),
        ]

    legend_handles.append(
        Line2D(
            [0], [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Selected checkpoint",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )

    filename = (
        f"validation_{metric}"
        if validation_only
        else f"training_{metric}"
    )

    for extension in ("png", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"{filename}.{extension}",
            dpi=200,
            bbox_inches="tight",
        )

    plt.close(fig)


def print_summary(histories):
    """Report accuracy at the selected checkpoint."""
    print(
        f"{'Dataset':<10} "
        f"{'Model':<15} "
        f"{'Epoch':>6} "
        f"{'Val Acc':>10}"
    )

    for dataset in DATASETS:
        for model, label in MODELS.items():
            history = histories[dataset][model]
            best_index = best_epoch_index(history)

            accuracy = (
                history["validation_accuracy"][best_index] * 100
            )

            print(
                f"{dataset:<10} "
                f"{label:<15} "
                f"{best_index + 1:>6} "
                f"{accuracy:>9.2f}%"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Plot Leaffliction training histories"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Root checkpoint directory",
    )
    args = parser.parse_args()

    histories = load_histories(args.input_dir)

    plot_metric(
        histories,
        metric="accuracy",
        validation_only=True,
    )

    plot_metric(
        histories,
        metric="loss",
        validation_only=False,
    )

    print_summary(histories)


if __name__ == "__main__":
    main()
