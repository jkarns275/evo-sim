import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

AX_LABEL_SIZE = 16
TITLE_SIZE = 18
TICK_LABEL_SIZE = 14

def plot(with_co, without_co, title, output):
  file1 = with_co
  file2 = without_co # "build/modulate_a_eq_b_no_co.csv"
  
  data1 = pd.read_csv(file1)
  data2 = pd.read_csv(file2)
  
  x_data1 = data1["Variance"]
  y_data1 = data1["Converged Percentage"]
  ci1 = data1["95% CI"]
  
  x_data2 = data2["Variance"]
  y_data2 = data2["Converged Percentage"]
  ci2 = data2["95% CI"]
  
  y_upper1 = y_data1 + ci1
  y_lower1 = y_data1 - ci1
  
  y_upper2 = y_data2 + ci2
  y_lower2 = y_data2 - ci2
  
  plt.figure(figsize=(8, 6))
  
  sns.scatterplot(
      x=x_data1,
      y=y_data1,
      label="w/ Crossover",
      color="blue",
      edgecolor="k"
  )
  
  plt.fill_between(
      x_data1,
      y_lower1,
      y_upper1,
      color="blue",
      alpha=0.2,
      label="w/ Crossove 95% CI"
  )
  
  plt.plot(x_data1, y_upper1, color="blue", linestyle="--", linewidth=1)
  plt.plot(x_data1, y_lower1, color="blue", linestyle="--", linewidth=1)
  
  sns.scatterplot(
      x=x_data2,
      y=y_data2,
      label="No Crossover",
      color="green",
      edgecolor="k"
  )
  
  plt.fill_between(
      x_data2,
      y_lower2,
      y_upper2,
      color="green",
      alpha=0.2,
      label="No Crossover 95% CI"
  )
  
  plt.plot(x_data2, y_upper2, color="green", linestyle="--", linewidth=1)
  plt.plot(x_data2, y_lower2, color="green", linestyle="--", linewidth=1)
  
  plt.title(title, fontsize=TITLE_SIZE)
  plt.xlabel("$\sigma^2$", fontsize=AX_LABEL_SIZE)
  plt.ylabel("p(R = 1)", fontsize=AX_LABEL_SIZE)
  plt.legend(fontsize=AX_LABEL_SIZE)
  plt.grid(True)
  plt.tight_layout()
  
  plt.savefig(output)
  plt.close()

plot("build/modulate_a_eq_b_co.csv", "build/modulate_a_eq_b_no_co.csv", "Convergence Probability vs. Initialization Radius; A = B", "plots/crossover_initialization_a_eq_b.pdf")
plot("build/modulate_a_lt_b_co.csv", "build/modulate_a_lt_b_no_co.csv", "Convergence Probability vs. Initialization Radius; 1.5A = B", "plots/crossover_initialization_a_lt_b.pdf")
