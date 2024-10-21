import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Data extracted from the table in the image for the "Average" row
labels = ['Qwen-zs', 'Qwen-zs-tip', 'Qwen-fs', 'Qwen-fs-tip', 'DeepSeek-zs', 'DeepSeek-zs-tip', 'DeepSeek-fs', 'DeepSeek-fs-tip','GPT-4o mini-zs', 'GPT-4o mini-zs-tip', 'GPT-4o mini-fs', 'GPT-4o mini-fs-tip', 'GPT-4o-zs', 'GPT-4o-zs-tip', 'GPT-4o-fs', 'GPT-4o-fs-tip']
# labels = ['Qwen-zs', 'Qwen-zs-tip', 'Qwen-fs', 'Qwen-fs-tip', 'DeepSeek-zs', 'DeepSeek-zs-tip', 'DeepSeek-fs', 'DeepSeek-fs-tip','4o mini-zs', '4o mini-zs-tip', '4o mini-fs', '4o mini-fs-tip', '4o-zs', '4o-zs-tip', '4o-fs', '4o-fs-tip']

average_values = [-3.0, 0.2, 0.2, 4.6, -2.6, 1.0, 2.0, 5.0, -0.4, 0.2, 3.0, 3.0, -0.2, 3.0, -1.6, 0.2]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define color mapping for zs, fs, zs-tip, and fs-tip configurations
color_mapping = {
    'zs': '#4A90E2',
    'zs-tip': '#F5A623',
    'fs': '#7ED321',
    'fs-tip': '#FF6F61'
}

# Assign colors to each configuration based on their type
bar_colors = []
for label in labels:
    if 'zs-tip' in label:
        bar_colors.append(color_mapping['zs-tip'])
    elif 'fs-tip' in label:
        bar_colors.append(color_mapping['fs-tip'])
    elif 'zs' in label:
        bar_colors.append(color_mapping['zs'])
    elif 'fs' in label:
        bar_colors.append(color_mapping['fs'])

# Plot the bar chart with the assigned colors
bars = ax.bar(labels, average_values, color=bar_colors)

# Add baseline lines for the specified AIs using the same color but different line styles
baselines = {
    'coacAI': 4,
    'workerRushAI': 1.8,
    'naiveMCTSAI': 0.2,
    'lightRushAI': -2.6,
    'randomBiasedAI': -3.4,
}
baseline_color = ['blue', 'blue', 'black', 'black', 'black']
line_styles = ['--', '-.', ':',  (0, (3, 4, 1, 4)), (0, (5, 5))]
baseline_color.reverse()
line_styles.reverse()

# Plot baselines with the same color but different line styles
for (ai, val), style, clr in zip(baselines.items(), line_styles, baseline_color):
    ax.axhline(y=val, linestyle=style, color=clr, label=f'{ai} ({val})')
    # ax.text(len(labels) - 1, val, f'{ai} ({val})', va='center', ha='left', color=baseline_color, fontsize=9, bbox=dict(facecolor='white', alpha=0.6))

# Add labels and title
# ax.set_xlabel('Configurations')
ax.set_ylabel('Average Scores', fontsize=14)
ax.set_title('Average Scores Against Different LLM-based Agents with Four PLAP Methods.', fontsize=16)
ax.set_ylim([-3.6, 5.6])
ax.bar_label(bars,fontsize=14, padding=1)
ax.tick_params(axis='y', direction='in')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=11, loc='upper right', ncol = 2)
# plt.legend(fontsize=11, bbox_to_anchor=(0.84, 0.31), ncol = 2)

# Display the chart
plt.tight_layout()
plt.savefig('exp2.pdf')
