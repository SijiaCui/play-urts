import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['Qwen2\n72B', 'DeepSeek\nV2.5', 'Claude 3\nHaiku', 'Claude 3.5\nSonnet', 
        'Gemini 1.5\nFlash', 'GPT-3.5\nTurbo', 'GPT-4o\nmini', 'GPT-4o', 'The Average']
models = ['Qwen2-72B', 'DeepSeek V2.5', 'Claude 3 Haiku', 'Claude 3.5 Sonnet', 
        'Gemini 1.5 Flash', 'GPT-3.5 Turbo', 'GPT-4o mini', 'GPT-4o', 'The Average']

# MicroRTS and StarcraftII QA correct rate
microRTS_scores = [0, 0, 0, 1, 0, 0, 0, 0.2, 0.15]
starcraftII_scores = [0.4, 1, 0.8, 0.8, 0.6, 1, 0.8, 1, 0.8]

# Create a horizontal bar plot
fig, ax = plt.subplots()

# Bar positions for the two categories
bar_width = 0.44
# positions_microRTS = range(len(models))
# positions_starcraftII = [pos + bar_width for pos in positions_microRTS]
positions_starcraftII = range(len(models))
positions_microRTS = [pos + bar_width for pos in positions_starcraftII]

# Plot the data
bars1 = ax.barh(positions_starcraftII, starcraftII_scores, height=bar_width, label='Starcraft II', color='#F5A623')
bars2 = ax.barh(positions_microRTS, microRTS_scores, height=bar_width, label='MicroRTS', color='#4A90E2')

ax.set_yticks([pos + bar_width/2 for pos in positions_starcraftII])
ax.set_yticklabels(models, fontsize=13)
ax.set_xlim(0,1.05)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis='x', labelsize=14)
ax.set_xlabel('Accuracy Rates', fontsize=14)
ax.set_title("Question-Answering Accuracy Rates on MicroRTS and \nStarCraft II Against Different LLMs.", fontsize=12)
ax.bar_label(bars1,fontsize=12, padding=1)
ax.bar_label(bars2,fontsize=12, padding=1)

# plt.yticks(rotation=30,fontsize=12)
plt.legend(fontsize=14, bbox_to_anchor=(0.94, 0.23))
plt.tight_layout()

plt.savefig("exp1.pdf")
