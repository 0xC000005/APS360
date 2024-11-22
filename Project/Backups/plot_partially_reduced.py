import matplotlib.pyplot as plt
import numpy as np

# Data
movements = [
    "Art_Nouveau_Modern",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Impressionism",
    "Post_Impressionism",
    "Realism",
    "Romanticism"
]

# Original (unbalanced) counts and percentages
original_counts = [4334, 2235, 1391, 6736, 13060, 6450, 10733, 7019]
original_percentages = [8.34, 4.30, 2.68, 12.96, 25.14, 12.41, 20.66, 13.51]

# Balanced counts and percentages
balanced_counts = [1391] * 8
balanced_percentages = [12.50] * 8

# Create figure and axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot absolute numbers
y_pos = np.arange(len(movements))
width = 0.35

# Original distribution bars
bars1_orig = ax1.barh(y_pos - width/2, original_counts, width,
                      label='Original', color='#1f77b4')
# Balanced distribution bars
bars1_bal = ax1.barh(y_pos + width/2, balanced_counts, width,
                     label='Balanced', color='#2ca02c')

ax1.set_xlabel('Number of Images')
ax1.set_title('Art Movement Distribution - Absolute Numbers')
ax1.set_yticks(y_pos)
ax1.set_yticklabels([m.replace('_', ' ') for m in movements])
ax1.legend()

# Add value labels on the bars
def autolabel_h(bars, ax):
    for rect in bars:
        width = rect.get_width()
        ax.text(width + 100, rect.get_y() + rect.get_height()/2,
                f'{int(width):,}',
                ha='left', va='center')

autolabel_h(bars1_orig, ax1)
autolabel_h(bars1_bal, ax1)

# Plot percentages
bars2_orig = ax2.barh(y_pos - width/2, original_percentages, width,
                      label='Original', color='#1f77b4')
bars2_bal = ax2.barh(y_pos + width/2, balanced_percentages, width,
                     label='Balanced', color='#2ca02c')

ax2.set_xlabel('Percentage of Dataset')
ax2.set_title('Art Movement Distribution - Percentages')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([m.replace('_', ' ') for m in movements])
ax2.legend()

# Add percentage labels on the bars
def autolabel_h_percent(bars, ax):
    for rect in bars:
        width = rect.get_width()
        ax.text(width + 0.5, rect.get_y() + rect.get_height()/2,
                f'{width:.2f}%',
                ha='left', va='center')

autolabel_h_percent(bars2_orig, ax2)
autolabel_h_percent(bars2_bal, ax2)

# Add grid lines
ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
ax2.grid(True, axis='x', linestyle='--', alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.savefig('distribution_comparison.png')
plt.close()

if __name__ == '__main__':
    pass