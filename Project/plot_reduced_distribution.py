import matplotlib.pyplot as plt
import numpy as np

# Data
classes = ['Expressionism', 'Impressionism', 'Post_Impressionism', 'Realism', 'Romanticism']
original_counts = [6736, 13060, 6450, 10733, 7019]
balanced_counts = [6450, 6450, 6450, 6450, 6450]

# Calculate percentages
original_total = sum(original_counts)
balanced_total = sum(balanced_counts)
original_percentages = [count/original_total * 100 for count in original_counts]
balanced_percentages = [count/balanced_total * 100 for count in balanced_counts]

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot absolute numbers
x = np.arange(len(classes))
width = 0.35

rects1 = ax1.bar(x - width/2, original_counts, width, label='Original', color='#1f77b4')
rects2 = ax1.bar(x + width/2, balanced_counts, width, label='Balanced', color='#2ca02c')

ax1.set_ylabel('Number of Images')
ax1.set_title('Dataset Distribution - Absolute Numbers')
ax1.set_xticks(x)
ax1.set_xticklabels(classes, rotation=45, ha='right')
ax1.legend()

# Add value labels on the bars
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1, ax1)
autolabel(rects2, ax1)

# Plot percentages
rects3 = ax2.bar(x - width/2, original_percentages, width, label='Original', color='#1f77b4')
rects4 = ax2.bar(x + width/2, balanced_percentages, width, label='Balanced', color='#2ca02c')

ax2.set_ylabel('Percentage of Dataset')
ax2.set_title('Dataset Distribution - Percentages')
ax2.set_xticks(x)
ax2.set_xticklabels(classes, rotation=45, ha='right')
ax2.legend()

# Add percentage labels on the bars
def autolabel_percentage(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel_percentage(rects3, ax2)
autolabel_percentage(rects4, ax2)

# Adjust layout and display
plt.tight_layout()
plt.show()
plt.savefig('dataset_distribution.png')
plt.close()

if __name__ == '__main__':
    pass