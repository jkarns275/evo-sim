from scipy.optimize import minimize
from scipy.optimize import curve_fit

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend("pdf", FigureCanvasPgf)

repro_a_eq_b = "repro_a_eq_b.csv"
repro_a_eq_b_name_map = {
    "Const(1)": "$t(x) = 1.0$",
    "Scott-DeJong( 10.00; 10.00 )": "$t(x) = f_{a, b}(x)$",
    "Scott-DeJong( 10.00; -10.00 )": "$t(x) = f_{a, -b}(x)$",
}

repro_b_gt_a = "repro_b_gt_a.csv"
repro_b_gt_a_name_map = {
    "Const(1)": "$t(x) = 1.0$",
    "Scott-DeJong( 10.00; 15.00 )": "$t(x) = f_{a, b}(x)$",
    "Scott-DeJong( -10.00; 15.00 )": "$t(x) = f_{-a, b}(x)$",
    "Scott-DeJong( 10.00; -15.00 )": "$t(x) = f_{a, -b}(x)$",
}

extended_c_xd = "extended_c_xd.csv"
extended_c_neg_xd = "extended_c_neg_xd.csv"

AX_LABEL_SIZE = 16
TITLE_SIZE = 18
TICK_LABEL_SIZE = 14

def make_bar_chart(csv_file: str, plot_title: str, x_label: str, xticks: dict, y_label: str, output_file: str, inverse: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    df = pd.read_csv(csv_file)
    
    converged_percentage = df["Converged Percentage"]
    if inverse:
        converged_percentage = 100.0 - converged_percentage
    confidence_interval = df["95% CI"]
    
    ax.bar(df.index, converged_percentage, yerr=confidence_interval, capsize=5, color="skyblue", edgecolor="black")
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)

    ax.set_xticks(df.index, df["Experiment Name"].map(xticks))

    fig.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)

make_bar_chart(repro_a_eq_b, "Convergence Probability w/ A = B", "", repro_a_eq_b_name_map, "p(A)", "plots/repro_a_eq_b.pdf")
make_bar_chart(repro_b_gt_a, "Convergence Probability w/ B = 1.5A", "", repro_b_gt_a_name_map, "p(A)", "plots/repro_b_gt_a.pdf", True)

def plot_flat_fitness_scatter(output_file):
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.read_csv('repro_flat_fitness.csv')
    
    x0 = np.array(df['x0'])
    x1 = np.array(df['x1'])
    
    print(f"N({np.mean(x0)}, {np.std(x0)})")
    print(f"N({np.mean(x1)}, {np.std(x1)})")
    
    duration = df['Duration']
    
    plt.figure(figsize=(8, 6))
    
    plt.grid(True)
    scatter = plt.scatter(x0, x1, c=duration, cmap='viridis', edgecolor='k')
    
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Duration', fontsize=TICK_LABEL_SIZE)
    
    ax.set_xlabel('$x_0$', fontsize=AX_LABEL_SIZE)
    ax.set_ylabel('$x_1$', fontsize=AX_LABEL_SIZE)
    ax.set_title('Random Individuals from 5000 Runs w/ Flat Fitness', fontsize=TITLE_SIZE)
    ax.text(5, 5, 'A', fontsize=24, ha='center', va='center', color='black')
    ax.text(-5, -5, 'B', fontsize=24, ha='center', va='center', color='black')
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_axisbelow(True)
    

    fig.tight_layout()
    fig.savefig(output_file)

plot_flat_fitness_scatter("plots/repro_flat_fitness.pdf")

def plot_wall_clock(output_file):
    df = pd.read_csv('repro_wall_clock_seq.csv')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for index, row in list(df.iterrows()):
        start = row['Start']
        duration = row['Duration']
        end = start + duration
        y_value = index  # Place each line on a successive Y value
        ax.hlines(y=y_value, xmin=start, xmax=end, color='black', linewidth=3)
    
    ax.axvline(x=723.2, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=768.5, color='red', linestyle=':', alpha=0.5)
    
    ax.set_ylim(300, 400)
    ax.set_xlim(710, 780)
    
    ax.set_xlabel('Time', fontsize=AX_LABEL_SIZE)
    ax.set_ylabel('Birth Step', fontsize=AX_LABEL_SIZE)
    ax.set_title('Evaluation Sequence in Wall-Clock Time', fontsize=TITLE_SIZE)
    ax.grid(True)

    fig.tight_layout()
    
    plt.savefig(output_file)

plot_wall_clock("repro_wall_clock_seq.pdf")
