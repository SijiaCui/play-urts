import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['Qwen2\n72B', 'DeepSeek\nV2.5', 'Claude 3\nHaiku', 'Claude 3.5\nSonnet', 
        'Gemini 1.5\nFlash', 'GPT-3.5\nTurbo', 'GPT-4o\nmini', 'GPT-4o']

# MicroRTS and StarcraftII QA correct rate
microRTS_scores = [0, 0, 0, 1, 0, 0, 0, 0.2]
starcraftII_scores = [0.4, 1, 0.8, 0.8, 0.6, 1, 0.8, 1]

# Create a horizontal bar plot
fig, ax = plt.subplots()

# Bar positions for the two categories
bar_width = 0.4
positions_microRTS = range(len(models))
positions_starcraftII = [pos + bar_width for pos in positions_microRTS]

# Plot the data
plt.barh(positions_microRTS, microRTS_scores, height=bar_width, label='MicroRTS', color='blue')
plt.barh(positions_starcraftII, starcraftII_scores, height=bar_width, label='Starcraft II', color='green')

ax.set_yticks([pos + bar_width/2 for pos in positions_microRTS])
ax.set_yticklabels(models)
ax.set_xlim(0,1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel('Accuracy Rates')
ax.set_title("Question-Answering Accuracy Rates on MicroRTS and StarCraft II \nagainst different LLMs.")

plt.legend()
plt.tight_layout()

plt.savefig("exp0.pdf")
