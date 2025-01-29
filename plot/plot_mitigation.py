import os
import pandas as pd
import matplotlib.pyplot as plt
import math

from plot_config import *

groups = [
    [],
    ["GEN"],
    ["SWT"],
    ["CNT"],
]

def file_names():
    return ["results/mitigation_control.csv"] + ["results/mitigation_" + "+".join(y.lower() for y in x) + ".csv" for x in groups[1:]]
def pretty_names():
    return ["Control"] + ["+".join(x) for x in groups[1:]]

output_plot_path = "figures/grid_plot.pdf"

def plot_csv_data(axes, data, title, color):
    print(data)
    for i, ax in enumerate(axes):
        categories = data["Experiment Name"][i * 4:4 + i * 4]
        values = data["Converged Percentage"][i * 4:4 + i * 4]
        ci = data["95% CI"][i * 4:4 + i * 4]
        hbars = ax.bar(list(range(4)), values, yerr=ci, color=color)
        ax.bar_label(hbars, fmt="%.0f", fontsize=BAR_LABEL_SIZE - 2)
        # ax.set_xticks(categories)
        # ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=6)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

def create_grid_plot():
    csv_files = file_names()
    names = pretty_names()
    
    num_files = len(csv_files)
    num_cols = 5
    num_rows = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8.27, 11.69 / 2), dpi=100, sharex=True, sharey=True)
    
    for i, csv_file in enumerate(csv_files):
        data = pd.read_csv(csv_file)
        a = axes[i]
        plot_csv_data(a, data, csv_file, colors[i])

    pad = 5 # in points
    column_names = ["$f_{A, B}; A = B$", "$f_{A, B}; 1.5A = B$", "$f_{A, B}; 2A = B$",
                    "$f_{A, B}; 2.5A = B$", "$f_{A, B}; 3A = B$"]
    for ax, col in zip(axes[0], column_names):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords="axes fraction", textcoords="offset points",
                    size="large", ha="center", va="baseline", weight="bold")
    
    for ax, row in zip(axes[:,0], names):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords="offset points",
                    size=AX_LABEL_SIZE, ha="right", va="center")
        ax.set_ylabel("p(R = 1)", fontsize=AX_LABEL_SIZE)
        ax.set_ylim(0, 100.0)
        [x.set_fontsize(MINOR_TICK_LABEL_SIZE) for x in ax.get_yticklabels()]
    print(axes[-1, :])
    for ax in axes[-1,:]:
        ax.set_xticks(list(range(4)))
        ax.set_xticklabels(["$f_{-A, B}$", "1", "$f_{A, B}$", "$f_{A, -B}$"], rotation=45, ha="right", rotation_mode="anchor", fontsize=MINOR_TICK_LABEL_SIZE)
        ax.set_xlabel("$t(x)$", fontsize=AX_LABEL_SIZE)

    fig.suptitle("Grid of Converged Percentages", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.98)
    plt.savefig(output_plot_path, dpi=300)
    plt.show()

create_grid_plot()
