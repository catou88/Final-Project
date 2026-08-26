"""Create a four-panel comparison chart from the reported regression metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Values extracted from report_model_comparison.html.
metrics = pd.DataFrame(
    {
        "Model": [
            "Linear\nregression",
            "Histogram gradient\nboosting",
            "CatBoost categorical\nmodel",
        ],
        "MAE": [0.0854, 0.0734, 0.0766],
        "RMSE": [0.1195, 0.1093, 0.1111],
        "R²": [0.8149, 0.8451, 0.8399],
        "Within 10 percentage points": [0.7084, 0.7493, 0.7338],
    }
).set_index("Model")


COLORS = ["#F59E0B", "#2563EB", "#8B5CF6"]


def add_value_labels(ax, values, percentage=False):
    """Place readable values above each bar."""
    offset = (max(values) - min(values)) * 0.06 or max(values) * 0.02

    for bar, value in zip(ax.patches, values):
        label = f"{value * 100:.2f}%" if percentage else f"{value:.4f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


fig, axes = plt.subplots(2, 2, figsize=(14, 9))

fig.suptitle(
    "Next-Hour Parking Availability: Model Comparison",
    fontsize=17,
    fontweight="bold",
)

chart_settings = [
    ("MAE", "MAE (occupancy proportion)", "Lower is better", False, "min"),
    ("RMSE", "RMSE (occupancy proportion)", "Lower is better", False, "min"),
    ("R²", "R² score", "Higher is better", False, "max"),
    (
        "Within 10 percentage points",
        "Predictions within ±10 percentage points",
        "Higher is better",
        False,
        "max",
    ),
]


for ax, (
    metric,
    ylabel,
    direction,
    as_percentage,
    best_rule,
) in zip(axes.flat, chart_settings):

    values = metrics[metric]

    bars = ax.bar(
        metrics.index,
        values,
        color=COLORS,
        width=0.68,
    )

    best_model = (
        values.idxmin()
        if best_rule == "min"
        else values.idxmax()
    )

    best_index = list(values.index).index(best_model)

    bars[best_index].set_edgecolor("#111827")
    bars[best_index].set_linewidth(3)

    add_value_labels(
        ax,
        values,
        percentage=as_percentage,
    )

    ax.set_title(
        f"{metric} ({direction})",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # Start the y-axis at zero to avoid exaggerating differences between models.
    upper = values.max() * 1.18
    ax.set_ylim(0, upper)
    
    ax.tick_params(axis="x", labelsize=9)


fig.text(
    0.5,
    0.015,
    "Black outline indicates the best-performing model for each metric.",
    ha="center",
    fontsize=10,
    color="#374151",
)

plt.tight_layout(rect=[0, 0.04, 1, 0.94])


# Save the image inside Final-Project/data.
output_directory = (
    Path.home()
    / "INDENG210"
    / "group_project"
    / "Final-Project"
    / "output"
)

# Create the data folder if it does not already exist.
output_directory.mkdir(parents=True, exist_ok=True)

output_path = output_directory / "model_metrics_comparison4.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"PNG image saved to: {output_path}")

plt.show()