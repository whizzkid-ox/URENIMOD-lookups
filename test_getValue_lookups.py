# Author: Ryo Segawa (whizznihil.kid@gmail.com)

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Directory containing pickle files
pkl_dir = r"C:\Users\rsegawa\URENIMOD\proposal_1\scripts\lookups\NME\lookup_tables\unmyelinated_fiber_20250527"

# Get all pkl files in directory
pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]

if not pkl_files:
    print(f"No pkl files found in {pkl_dir}")
    exit(1)

# Select a random pkl file
pkl_file = os.path.join(pkl_dir, random.choice(pkl_files))
print(f"\nSelected file: {pkl_file}\n")

# Read the pickle file
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# Print reference values in a formatted way
print("\nReference Values:")
print("=" * 50)
for key, value in data['refs'].items():
    unit = ""
    if key == 'D0' or key == 'd0' or key == 'l0':
        unit = "m"
    elif key == 'f':
        unit = "Hz"
    elif key == 'A':
        unit = "Pa"
    value_float = float(value[0])
    print(f"{key:4s}: {value_float:10.3e} {unit}")

# Print table values in a formatted way
print("\nTable Values:")
print("=" * 50)
for key, value in data['tables'].items():
    value_float = float(value.flatten()[0])
    print(f"{key:10s}: {value_float:15.6e}")

# Create figure with subplots
plt.figure(figsize=(15, 10))

# Plot 1: Reference Values
plt.subplot(2, 1, 1)
ref_keys = list(data['refs'].keys())
ref_values = [float(data['refs'][k][0]) for k in ref_keys]
colors = ['b' if v >= 0 else 'r' for v in ref_values]

# Use scientific notation for y-axis
plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
plt.bar(ref_keys, ref_values, color=colors)
plt.title('Reference Values')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.grid(True)

# Plot 2: Group table values by magnitude
plt.subplot(2, 1, 2)
table_data = {
    'Large (>1)': [],
    'Medium (1e-3 to 1)': [],
    'Small (<1e-3)': [],
    'Zero': []
}

table_keys = {
    'Large (>1)': [],
    'Medium (1e-3 to 1)': [],
    'Small (<1e-3)': [],
    'Zero': []
}

for key, value in data['tables'].items():
    val = abs(float(value.flatten()[0]))
    if val == 0:
        table_data['Zero'].append(float(value.flatten()[0]))
        table_keys['Zero'].append(key)
    elif val > 1:
        table_data['Large (>1)'].append(float(value.flatten()[0]))
        table_keys['Large (>1)'].append(key)
    elif val >= 1e-3:
        table_data['Medium (1e-3 to 1)'].append(float(value.flatten()[0]))
        table_keys['Medium (1e-3 to 1)'].append(key)
    else:
        table_data['Small (<1e-3)'].append(float(value.flatten()[0]))
        table_keys['Small (<1e-3)'].append(key)

# Create subplots for each magnitude group
fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
fig2.suptitle('Table Values Grouped by Magnitude', fontsize=16)

for (i, (group, values)), ax in zip(enumerate(table_data.items()), axes.flatten()):
    if values:
        colors = ['b' if v >= 0 else 'r' for v in values]
        ax.bar(table_keys[group], values, color=colors)
        ax.set_title(group)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        # Use scientific notation for y-axis
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))

plt.tight_layout()
plt.show()
