import pandas as pd
from pycirclize import Circos
import matplotlib.pyplot as plt


data = {
    "LLM": [
        "Qwen2-72B-Instruct",
        "Claude 3.5 Sonnet",
        "Claude 3 Haiku",
        "DeepSeek V2.5",
        "Gemini 1.5 Flash",
        "GPT-3.5 Turbo",
        "GPT-4o",
        "GPT-4o mini",
    ],
    "Unit Production": [6.650, 16.150, 16.950, 10.725, 15.375, 15.425, 12.375, 16.300],
    "Unit Kills": [7.875, 15.725, 13.075, 7.975, 16.050, 14.475, 11.825, 13.300],
    "Unit Losses": [4.675, 15.475, 13.875, 8.425, 14.975, 14.750, 8.650, 14.775],
    "Damage Dealt": [13.125, 21.475, 16.300, 10.150, 22.975, 18.475, 19.475, 16.625],
    "Damage Taken": [10.450, 18.775, 18.275, 14.450, 18.775, 19.525, 11.950, 20.550],
    "Resources Spent": [10.200, 16.950, 17.250, 14.125, 15.375, 16.125, 18.000, 16.950],
    "Resources Harvested": [
        11.925,
        19.575,
        13.850,
        12.675,
        18.875,
        13.225,
        17.875,
        14.400,
    ],
    "Game Time": [
        738.950,
        1252.400,
        1407.000,
        1295.650,
        978.400,
        1169.400,
        768.875,
        1337.150,
    ],
}

df = pd.DataFrame.from_dict(data).set_index("LLM")

# df = df.drop(index="GPT-3.5 Turbo")
# df = df.drop(index="GPT-4o mini")
# df = df.drop(index="Claude 3 Haiku")

df["Kill/Death Ratio"] = df["Unit Kills"] / df["Unit Losses"]
df["Resource Efficiency"] = df["Damage Dealt"] / df["Resources Spent"]
df["Damage Efficiency"] = df["Damage Dealt"] / df["Damage Taken"]
df["Unit Production Efficiency"] = df["Unit Kills"] / df["Unit Production"]
df["Resource Harvesting Efficiency"] = (df["Resources Harvested"] / df["Game Time"]) * 100
df["Unit Combat Efficiency"] = df["Damage Dealt"] / df["Unit Kills"]

df.loc["Average"] = df.mean()

colors = [
    "#A8DADC",  # 淡蓝绿
    "#457B9D",  # 深蓝
    "#C7EFCF",  # 淡绿
    "#E63946",  # 鲜艳红
    "#6D597A",  # 紫灰色
    "#B56576",  # 浅紫红
    "#FFDDD2",  # 淡粉红
    "#FFE066",  # 柠檬黄
]
color_map = {k: v for k, v in zip(df.index[:-1], colors)}
color_map.update({"Average": "#D3D3D3"})

print(color_map)

fig = plt.figure()
fig.subplots(2, 4, subplot_kw=dict(polar=True))
plt.subplots_adjust(wspace=0.6, hspace=0.2)
for index, ax in zip(df.index[:-1], fig.axes):
    d = pd.DataFrame([df.loc[index], df.loc["Average"]], index=[index, "Average"]).iloc[:, 8:]
    circos = Circos.radar_chart(
        d,
        vmax=2.5,
        bg_color=None,
        cmap=color_map,
        show_grid_label=False,
        grid_line_kws=dict(lw=0.2, ls="--"),
        line_kws_handler=lambda _: dict(lw=1, ls="-"),
        label_kws_handler=lambda _: dict(size=5),
    )
    circos.plotfig(ax=ax)
fig.savefig("radar2.pdf")




# circos = Circos.radar_chart(
#     df.iloc[:-1, 8:],
#     vmax=d.max().max(),
#     grid_interval_ratio=0.2,
#     show_grid_label=False,
#     bg_color=None,
#     grid_line_kws=dict(lw=0.5, ls="--"),
#     line_kws_handler=lambda _: dict(lw=3, ls="-"),
#     label_kws_handler=lambda _: dict(size=10),
# )

# fig = circos.plotfig()
# fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=5)
# fig.savefig("radar.pdf")
