import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend("pdf", FigureCanvasPgf)

df = pd.read_csv('build/repro_wall_clock_seq.csv')

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

ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Birth Step', fontsize=14)
ax.set_title('Evaluation Sequence in Wall-Clock Time', fontsize=16)
ax.grid(True)
fig.tight_layout()

plt.savefig("repro_wall_clock_seq.pdf")
plt.show()
