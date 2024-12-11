import matplotlib.pyplot as plt

# Define a general style for all plots
plt.rcParams.update({
    'font.size': 16,          # Increase overall font size
    'axes.labelsize': 18,     # Increase axis label font size
    'axes.titlesize': 20,     # Increase title font size
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'legend.fontsize': 16,    # Increase legend font size
    'xtick.labelsize': 16,    # Increase x-tick label font size
    'ytick.labelsize': 16     # Increase y-tick label font size
})

# Sample data for the first plot, multiplied by 100
x = [val * 100 for val in [0.5075, 0.525, 0.5175, 0.505, 0.5025]]
y = [val * 100 for val in [0.5125, 0.515, 0.53, 0.5075, 0.4925]]

# Colors and languages for each pair of points
colors = ['red', 'blue', 'green', 'orange', 'purple']
languages = ['Danish', 'Arabic', 'Japanese', 'Hindi', 'English']

# Create figure for Accuracy Scatter Plot
plt.figure(figsize=(10, 6))  # Larger figure size for better readability

# Plot each language with a unique color and label
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], label=f'{languages[i]}', s=100, edgecolor='black', alpha=0.7)

# Add labels and title with increased size for scientific appearance
plt.xlabel('Tested on (%)', fontsize=16)
plt.ylabel('Trained on (%)', fontsize=16)
plt.title('Accuracy Scatter Plot', fontsize=18, weight='bold')

# Highlight the 50 line for both axes
plt.axhline(y=50, color='black', linestyle='--', linewidth=1.5)
plt.axvline(x=50, color='black', linestyle='--', linewidth=1.5)

# Add grid for better readability
plt.grid(True, linestyle='--', linewidth=0.7)

# Display the legend with unique labels in the upper left corner
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys(), title="Languages", title_fontsize='18', loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()


# Sample data for the second plot
c =  [0.505, 0.5425, 0.53, 0.505, 0.495]
d =  [0.52, 0.5175, 0.5425, 0.5125, 0.485]

# Create figure for RUC Plot
plt.figure(figsize=(10, 6))  # Larger figure size for better readability

# Plot each language with a unique color and label
for i in range(len(c)):
    plt.scatter(c[i], d[i], color=colors[i], label=f'{languages[i]}', s=100, edgecolor='black', marker='x', alpha=0.7)

# Add labels and title with increased size for scientific appearance
plt.xlabel('Tested on', fontsize=16)
plt.ylabel('Trained on', fontsize=16)
plt.title('AUC-RUC Plot', fontsize=18, weight='bold')

# Highlight the 50 line for both axes
plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5)
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5)

# Add grid for better readability
plt.grid(True, linestyle='--', linewidth=0.7)

# Display the legend with unique labels in the upper left corner
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys(), title="Languages", title_fontsize='18', loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()



