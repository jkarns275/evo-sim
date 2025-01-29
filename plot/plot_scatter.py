from scipy.optimize import minimize
from scipy.optimize import curve_fit

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load CSV file into a Pandas DataFrame
csv_file = 'build/results.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Specify the columns to plot
y_column = 'Converged Percentage'  # Replace with your desired column for the x-axis
x_column = ' Distance from Minimum'  # Replace with your desired column for the y-axis
y_error_column = " 95% CI"
# x_error_column = 'XErrorColumn'  # Replace with your 95% CI width for x
# Estimate standard errors for y (given as error values)
y_se = df[y_error_column]

# Define the linear model
def linear_model(x, a, b):
    return a * x + b

# Perform weighted least squares regression
popt, pcov = curve_fit(linear_model, df[x_column], df[y_column], sigma=y_se, absolute_sigma=True)

# Fit line parameters
a, b = popt

# Generate points for the regression line
x_fit = np.linspace(df[x_column].min(), df[x_column].max(), 100)
y_fit = linear_model(x_fit, a, b)

# Plotting
plt.errorbar(df[x_column], df[y_column], yerr=y_se, fmt='o', ecolor='red', capsize=3, label='Data points')
plt.plot(x_fit, y_fit, color='blue', label='Regression line')

# Add labels, title, and legend
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.title('Scatter Plot with Error Bars and Weighted Regression Line')
plt.legend()

# Show the plot
plt.show()
