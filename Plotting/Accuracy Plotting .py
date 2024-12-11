#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:07:56 2024

@author: ankadilfer
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a list of tuples with language training, testing, and accuracy data
data_flattened = [
    ("Danish", "Arabic", 0.52),
    ("Danish", "Japanese", 0.51),
    ("Danish", "Hindi", 0.52),
    ("Danish", "English", 0.50),
    ("Arabic", "Danish", 0.50),
    ("Arabic", "Japanese", 0.53),
    ("Arabic", "Hindi", 0.53),
    ("Arabic", "English", 0.50),
    ("Japanese", "Danish", 0.55),
    ("Japanese", "Arabic", 0.54),
    ("Japanese", "Hindi", 0.52),
    ("Japanese", "English", 0.51),
    ("Hindi", "Danish", 0.49),
    ("Hindi", "Arabic", 0.54),
    ("Hindi", "Japanese", 0.50),
    ("Hindi", "English", 0.50),
    ("English", "Danish", 0.49),
    ("English", "Arabic", 0.50),
    ("English", "Japanese", 0.53),
    ("English", "Hindi", 0.45)
]

# Create a DataFrame from the data
accuracy_df = pd.DataFrame(data_flattened, columns=["Trained on", "Tested on", "Accuracy"])
accuracy_df["Accuracy"] = accuracy_df["Accuracy"] * 100

# Set scientific plot style
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size': 20,          # Increase overall font size
    'axes.labelsize': 24,     # Increase axis label font size
    'axes.titlesize': 26,     # Increase title font size
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'legend.fontsize': 20,    # Increase legend font size
    'xtick.labelsize': 20,    # Increase x-tick label font size
    'ytick.labelsize': 20     # Increase y-tick label font size
})

# Filtering the DataFrame to remove rows where 'Trained on' is equal to 'Tested on'
filtered_accuracy_df = accuracy_df[accuracy_df['Trained on'] != accuracy_df['Tested on']]

# Get the unique 'Trained on' values that have valid data
trained_on_values = filtered_accuracy_df['Trained on'].unique()

# Set up the number of subplots needed based on valid 'Trained on' values
n_cols = len(trained_on_values)
fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), sharey=True)

# Define a list of palettes for each subplot (you can adjust this to your liking)
palettes = ['Set2', 'Blues', 'Greens', 'Reds', 'Purples']

# Create each subplot separately
for ax, trained_on_value, palette in zip(axes, trained_on_values, palettes):
    subset_df = filtered_accuracy_df[filtered_accuracy_df['Trained on'] == trained_on_value]
    sns.barplot(
        data=subset_df,
        x="Tested on",
        y="Accuracy",
        hue="Trained on",
        ax=ax,
        palette=palette  # Use a different palette for each plot
    )
    ax.set_title(f'Trained on {trained_on_value}')
    ax.set_ylabel('Accuracy' if ax == axes[0] else '')  # Add y-axis label only to the first subplot
    ax.set_xlabel('Tested on Language')
    ax.tick_params(axis='x')  # Rotate x-axis labels for better readability
    ax.legend_.remove()

# Adjust layout to ensure no overlaps and everything looks good
plt.tight_layout()
plt.show()
