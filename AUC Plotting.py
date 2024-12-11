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
roc_data_flattened = [
    ("Danish", "Arabic", 0.56),
    ("Danish", "Japanese", 0.53),
    ("Danish", "Hindi", 0.51),
    ("Danish", "English", 0.48),
    ("Arabic", "Danish", 0.48),
    ("Arabic", "Japanese", 0.54),
    ("Arabic", "Hindi", 0.55),
    ("Arabic", "English", 0.50),
    ("Japanese", "Danish", 0.56),
    ("Japanese", "Arabic", 0.56),
    ("Japanese", "Hindi", 0.53),
    ("Japanese", "English", 0.52),
    ("Hindi", "Danish", 0.50),
    ("Hindi", "Arabic", 0.57),
    ("Hindi", "Japanese", 0.50),
    ("Hindi", "English", 0.48),
    ("English", "Danish", 0.48),
    ("English", "Arabic", 0.48),
    ("English", "Japanese", 0.55),
    ("English", "Hindi", 0.43)
]

# Create a DataFrame from the data
accuracy_df = pd.DataFrame(roc_data_flattened, columns=["Trained on", "Tested on", "AU-ROC"])

# Set scientific plot style
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size': 18,          # Increase overall font size
    'axes.labelsize': 22,     # Increase axis label font size
    'axes.titlesize': 24,     # Increase title font size
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'legend.fontsize': 18,    # Increase legend font size
    'xtick.labelsize': 18,    # Increase x-tick label font size
    'ytick.labelsize': 18     # Increase y-tick label font size
})
# Filtering the DataFrame to remove rows where 'Trained on' is equal to 'Tested on'
filtered_accuracy_df = accuracy_df[accuracy_df['Trained on'] != accuracy_df['Tested on']]

# Get the unique 'Trained on' values that have valid data
trained_on_values = filtered_accuracy_df['Trained on'].unique()

# Set up the number of subplots needed based on valid 'Trained on' values
n_cols = len(trained_on_values)
fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)

# Define a list of palettes for each subplot (you can adjust this to your liking)
palettes = ['Set2', 'Blues', 'Greens', 'Reds', 'Purples']



# Create each subplot separately
for ax, trained_on_value, palette in zip(axes, trained_on_values, palettes):
    subset_df = filtered_accuracy_df[filtered_accuracy_df['Trained on'] == trained_on_value]
    sns.barplot(
        data=subset_df,
        x="Tested on",
        y="AU-ROC",
        hue="Trained on",
        ax=ax,
        palette=palette  # Use a different palette for each plot
    )
    
  
    
    ax.set_title(f'Trained on {trained_on_value}')
    ax.set_ylabel('AU-ROC' if ax == axes[0] else '')  # Add y-axis label only to the first subplot
    ax.set_xlabel('Tested on Language')
    ax.legend_.remove()

# Adjust layout to ensure no overlaps and everything looks good
plt.tight_layout()
plt.show()
