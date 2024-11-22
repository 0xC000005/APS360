import matplotlib.pyplot as plt
import numpy as np

# Data
movements = [
    "Abstract_Expressionism", "Action_painting", "Art_Nouveau_Modern", "Baroque",
    "Color_Field_Painting", "Contemporary_Realism", "Cubism", "Early_Renaissance",
    "Expressionism", "Fauvism", "High_Renaissance", "Impressionism",
    "Mannerism_Late_Renaissance", "Minimalism", "Naive_Art_Primitivism",
    "New_Realism", "Northern_Renaissance", "Pointillism", "Pop_Art",
    "Post_Impressionism", "Realism", "Rococo", "Romanticism", "Symbolism",
    "Synthetic_Cubism"
]

original_counts = [
    2782, 98, 4334, 4240, 1615, 481, 2235, 1391, 6736, 934, 1343, 13060,
    1279, 1337, 2405, 314, 2552, 513, 1483, 6450, 10733, 2089, 7019, 2546, 216
]

balanced_counts = [98] * 25  # All classes balanced to 98

# Calculate percentages
original_total = sum(original_counts)
balanced_total = sum(balanced_counts)
original_percentages = [count/original_total * 100 for count in original_counts]
balanced_percentages = [count/balanced_total * 100 for count in balanced_counts]

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

# Plot absolute numbers
x = np.arange(len(movements))
width = 0.35

rects1 = ax1.barh(x - width/2, original_counts, width, label='Original', color='#1f77b4')
rects2 = ax1.barh(x + width/2, balanced_counts, width, label='Balanced', color='#2ca02c')

ax1.set_xlabel('Number of Images')
ax1.set_title('Dataset Distribution - Absolute Numbers')
ax1.set_yticks(x)
ax1.set_yticklabels([m.replace('_', ' ') for m in movements])
ax1.legend()

# Add value labels on the bars
def autolabel(rects, ax):
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f'{int(width):,}',
                    xy=(width, rect.get_y() + rect.get_height()/2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center')

autolabel(rects1, ax1)
autolabel(rects2, ax1)

# Plot percentages
rects3 = ax2.barh(x - width/2, original_percentages, width, label='Original', color='#1f77b4')
rects4 = ax2.barh(x + width/2, balanced_percentages, width, label='Balanced', color='#2ca02c')

ax2.set_xlabel('Percentage of Dataset')
ax2.set_title('Dataset Distribution - Percentages')
ax2.set_yticks(x)
ax2.set_yticklabels([m.replace('_', ' ') for m in movements])
ax2.legend()

# Add percentage labels on the bars
def autolabel_percentage(rects, ax):
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f'{width:.1f}%',
                    xy=(width, rect.get_y() + rect.get_height()/2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center')

autolabel_percentage(rects3, ax2)
autolabel_percentage(rects4, ax2)

# Adjust layout and display
plt.tight_layout()
plt.show()
plt.savefig('full_dataset_distribution.png')
plt.close()

if __name__ == '__main__':
    pass