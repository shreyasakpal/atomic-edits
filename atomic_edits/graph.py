import matplotlib.pyplot as plt
import numpy as np

# Data from Table II - ATOMIC method
categories = ['Overall', 'Color', 'Remove', 'Property', 'Add']
atomic_scores = [58.3, 95.8, 25.4, 52.7, 48.9]
oneshot_scores = [40.2, 45.1, 18.3, 38.7, 42.5]  # for comparison

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Set positions
x = np.arange(len(categories))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, atomic_scores, width, label='ATOMIC', color='#2E86AB')
bars2 = ax.bar(x + width/2, oneshot_scores, width, label='One-shot', color='#A23B72')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Customize
ax.set_xlabel('Edit Type', fontsize=11)
ax.set_ylabel('Per-Requirement Accuracy (%)', fontsize=11)
ax.set_title('Success Rates Across Different Edit Types', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='upper right')
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Save
plt.tight_layout()
plt.savefig('success_rates_bar.png', dpi=300, bbox_inches='tight')
plt.show()