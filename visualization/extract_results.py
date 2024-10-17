import os
import numpy as np
import pandas as pd

NORM_MODEL_NAME_MAP = {
    "Qwen2-72B-Instruct": "Qwen2-72B-Instruct",
    "deepseek-chat": "DeepSeek V2.5",
    "gpt-4o-mini": "GPT-4o mini",
    "gpt-4o": "GPT-4o",
    "gemini-1.5-flash-002": "Gemini 1.5 Flash",
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
}

UNIT_PRODUCTION = 0
UNIT_KILLS = 1
UNIT_LOSSES = 2
DAMAGE_DEALT = 3
DAMAGE_TAKEN = 4
RESOURCES_SPENT = 5
RESOURCES_HARVESTED = 6


def extract_results(runs_dir, filename=None):
    """
    {
        'few_shot vs workerRushAI': {
            'point': -5
            'round 1': {
                'winner': ...,
                'detail': ...,
                'game_time': ...,
                'point': ...,
                'metrics': ...
            }
            'round 2': ...
        }
        'few_shot vs coacAI': ...
    }
    """
    results_list = {}
    matches = os.listdir(runs_dir)
    for match in matches:
        if not os.path.isdir(os.path.join(runs_dir, match)):
            continue
        match_name = match.replace("_vs_", " vs ")
        results_list[match_name] = {}
        point = 0  # total point of one match
        metrics = []  # avg metrics of all rounds in one match
        for run in os.listdir(os.path.join(runs_dir, match)):
            result = get_result(os.path.join(runs_dir, match, run))
            if result:
                results_list[match_name].update(result)
                point += (
                    list(result.values())[0]["point"]
                    if list(result.values())[0]["point"] is not None
                    else 0
                )
                results_list[match_name]["point"] = point
    if filename is not None:
        write_results(results_list, filename)
    return results_list


def write_results(results_list, filename):
    with open(filename, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 32 + "Matches Overview\n")
        f.write("=" * 80 + "\n\n")
        for match, results in sorted(results_list.items()):
            f.write(f"{match} -> {results.get('point', '')}\n")
            for round_index, result in sorted(results.items()):
                if "round" in round_index:
                    f.write(f"{round_index} winner: {result['winner']}\n")
            f.write("\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write(" " * 32 + "Matches Details\n")
        f.write("=" * 80 + "\n\n")
        for match, results in sorted(results_list.items()):
            f.write(match + "\n")
            for round_index, result in sorted(results.items()):
                if "round" in round_index:
                    f.write(f"{round_index} detail: \n{result['detail']}\n")


def get_result(run_dir):
    """
    {
        'round 2':
        {
            'winner': 'workerRushAI',
            'detail': 'Game over at 1334 step! The winner is workerRushAI...'},
            'game_time': 1334,
            'point': -1,
            'metrics': [17, 24, 19, 32, 31, 21, 19]
        }
    }
    """
    round_index = run_dir.split("_")[-1]
    result = {}
    log_path = os.path.join(run_dir, "run.log")
    if not os.path.exists(log_path):
        return result
    with open(log_path, "r") as f:
        text = f.read()
    if "Game over" in text:
        detail = "Game over" + text.split("Game over")[-1]
        winner = (
            detail.split("The winner is ")[-1].split("\n")[0]
            if "The winner is" in detail
            else "draw"
        )
        result[f"round {round_index}"] = {"winner": winner, "detail": detail}
        blue_metrics = filter(None, detail.split("Blue")[-1].split("\n")[0].split(" "))
        blue_metrics = [int(x) for x in blue_metrics]
        red_metrics = filter(None, detail.split("Red")[-1].split("\n")[0].split(" "))
        red_metrics = [int(x) for x in red_metrics]
        game_time = int(detail.split("Game over at ")[-1].split(" step")[0])
        point = None
        if winner == "draw":
            point = 0
        else:
            blue, red = run_dir.split("/")[-2].split("_vs_")
            if "blue" in winner or blue in winner:
                point = 1
            elif "red" in winner or red in winner:
                point = -1
        result[f"round {round_index}"]["game_time"] = game_time
        result[f"round {round_index}"]["point"] = point
        result[f"round {round_index}"]["metrics"] = {}
        result[f"round {round_index}"]["metrics"]["blue"] = blue_metrics
        result[f"round {round_index}"]["metrics"]["red"] = red_metrics
    return result


# def plot_metrics(runs_dir):
#     from pycirclize import Circos
#     import pandas as pd

#     player_metrics = get_avg_metrics(runs_dir)
#     data = {NORM_MODEL_NAME_MAP[k]: v for k, v in sorted(player_metrics.items())}

#     df = pd.DataFrame.from_dict(data, orient="index")
#     df.columns = [
#         "Unit Production",
#         "Unit Kills",
#         "Unit Losses",
#         "Damage Dealt",
#         "Damage Taken",
#         "Resources Spent",
#         "Resources Harvested",
#         "Game Time"
#     ]
#     print(df)

# Initialize Circos instance for radar chart plot
# circos = Circos.radar_chart(
#     df,
#     vmax=df.max().max(),
#     grid_interval_ratio=0.2,
#     show_grid_label=False,
#     bg_color=None,
#     grid_line_kws=dict(lw=0.5, ls="--"),
#     line_kws_handler=lambda _: dict(lw=2, ls="-"),
#     label_kws_handler=lambda _: dict(size=10),
# )

# fig = circos.plotfig()
# fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4)
# fig.savefig("radar.pdf")


# def get_avg_metrics(runs_dir: str) -> dict:
#     player_metrics = {}
#     results_list = extract_results(runs_dir)
#     for match, results in results_list.items():
#         if "o1-mini" in match:
#             continue
#         player = match.split(" vs ")[0]
#         if player not in player_metrics:
#             player_metrics[player] = []
#         player_metrics[player].append(results["metrics"])
#     player_avg_metrics = {}
#     for player, metrics in player_metrics.items():
#         player_avg_metrics[player] = np.average(metrics, axis=0)
#     return player_avg_metrics


def plot_metrics(runs_dir: str) -> dict:
    from pycirclize import Circos

    players_metrics = {}
    player_sides = ["blue", "red"]
    results_list = extract_results(runs_dir)
    for match, results in results_list.items():
        if "o1-mini" in match:
            continue
        players = match.split(" vs ")
        if players[0] == players[1]:
            continue
        for player, side in zip(players, player_sides):
            if player not in players_metrics:
                players_metrics[player] = {}
                players_metrics[player]["damage_dealt/damage_taken"] = []
                players_metrics[player]["resources_harvested/game_time"] = []
                players_metrics[player]["resources_spent/game_time"] = []
                players_metrics[player]["resources_spent/resources_harvested"] = []
                players_metrics[player]["unit_production/game_time"] = []
                players_metrics[player]["unit_production/resources_harvested"] = []
                players_metrics[player]["unit_kills/unit_losses"] = []
                players_metrics[player]["damage_dealt/unit_losses"] = []
                players_metrics[player]["unit_kills/unit_production"] = []
                players_metrics[player]["damage_dealt/resources_spent"] = []
                players_metrics[player][
                    "damage_dealt * resources_harvested / (damage_taken * game_time)"
                ] = []
                players_metrics[player][
                    "unit_production * resources_harvested / (game_time * game_time)"
                ] = []
            for round_index, round_result in results.items():
                if "round" in round_index:
                    metrics = round_result["metrics"][side]
                    game_time = round_result["game_time"] / 100
                    players_metrics[player]["resources_harvested/game_time"].append(
                        metrics[RESOURCES_HARVESTED] / game_time
                    )
                    players_metrics[player]["resources_spent/game_time"].append(
                        metrics[RESOURCES_SPENT] / game_time
                    )
                    players_metrics[player]["unit_production/game_time"].append(
                        metrics[UNIT_PRODUCTION] / game_time
                    )
                    players_metrics[player][
                        "unit_production * resources_harvested / (game_time * game_time)"
                    ].append(
                        metrics[UNIT_PRODUCTION]
                        * metrics[RESOURCES_HARVESTED]
                        / (game_time * game_time)
                    )
                    if metrics[DAMAGE_TAKEN] != 0:
                        players_metrics[player]["damage_dealt/damage_taken"].append(
                            metrics[DAMAGE_DEALT] / metrics[DAMAGE_TAKEN]
                        )
                        players_metrics[player][
                            "damage_dealt * resources_harvested / (damage_taken * game_time)"
                        ].append(
                            metrics[DAMAGE_DEALT]
                            * metrics[RESOURCES_HARVESTED]
                            / (metrics[DAMAGE_TAKEN] * game_time)
                        )
                    if metrics[RESOURCES_SPENT] != 0:
                        players_metrics[player]["damage_dealt/resources_spent"].append(
                            metrics[DAMAGE_DEALT] / metrics[RESOURCES_SPENT]
                        )
                    if metrics[RESOURCES_HARVESTED] != 0:
                        players_metrics[player][
                            "unit_production/resources_harvested"
                        ].append(
                            metrics[UNIT_PRODUCTION] / metrics[RESOURCES_HARVESTED]
                        )
                        players_metrics[player][
                            "resources_spent/resources_harvested"
                        ].append(
                            metrics[RESOURCES_SPENT] / metrics[RESOURCES_HARVESTED]
                        )
                    if metrics[UNIT_LOSSES] != 0:
                        players_metrics[player]["unit_kills/unit_losses"].append(
                            metrics[UNIT_KILLS] / metrics[UNIT_LOSSES]
                        )
                        players_metrics[player]["damage_dealt/unit_losses"].append(
                            metrics[DAMAGE_DEALT] / metrics[UNIT_LOSSES]
                        )
                    if metrics[UNIT_PRODUCTION] != 0:
                        players_metrics[player]["unit_kills/unit_production"].append(
                            metrics[UNIT_KILLS] / metrics[UNIT_PRODUCTION]
                        )

    for player, player_metrics in players_metrics.items():
        for metrics_name, metrics in player_metrics.items():
            players_metrics[player][metrics_name] = np.average(metrics)

    players_metrics.pop("gpt-3.5-turbo")
    players_metrics.pop("gpt-4o-mini")
    players_metrics.pop("claude-3-haiku-20240307")
    df = pd.DataFrame.from_dict(players_metrics, orient="index")

    df.drop(
        [
            "resources_spent/resources_harvested",
            "unit_production/resources_harvested",
            "damage_dealt/resources_spent",
            "unit_kills/unit_production",
            "damage_dealt/unit_losses",
            "unit_kills/unit_losses",
            # "damage_dealt * resources_harvested / (damage_taken * game_time)",
        ],
        axis=1,
        inplace=True,
    )

    pd.options.display.max_columns = None
    df.rename(
        columns={
            "damage_dealt/damage_taken": "CER",  # Combat Efficiency Ratio
            "resources_harvested/game_time": "RHE",  # Resource Harvesting Efficiency
            "resources_spent/game_time": "RUR",  # Resource Utilization Rate
            "unit_production/game_time": "UPR",  # Unit Production Rate
            "damage_dealt * resources_harvested / (damage_taken * game_time)": "CCER",  # Comprehensive Combat and Economy Ratio
            "unit_production * resources_harvested / (game_time * game_time)": "PESE",  # Production and Economic Synergy Efficiency
        },
        inplace=True,
    )

    df.index = df.index.str.replace(
        "gemini-1.5-flash-002", "Gemini 1.5 Flash", regex=False
    )
    df.index = df.index.str.replace("deepseek-chat", "DeepSeek V2.5", regex=False)
    df.index = df.index.str.replace("gpt-4o", "GPT-4o", regex=False)
    df.index = df.index.str.replace(
        "claude-3-5-sonnet-20240620", "Claude 3.5 Sonnet", regex=False
    )
    df.index = df.index.str.replace(
        "Qwen2-72B-Instruct", "Qwen2-72B-Instruct", regex=False
    )

    df["UPR"] = df["UPR"].apply(lambda x: x * 2)
    df["RHE"] = df["RHE"].apply(lambda x: x * 2)
    df["RUR"] = df["RUR"].apply(lambda x: x * 2)
    df["CCER"] = df["CCER"].apply(lambda x: x / 2)

    print(df)

    circos = Circos.radar_chart(
        df,
        vmax=4.97,
        grid_interval_ratio=0.2,
        show_grid_label=False,
        bg_color=None,
        grid_line_kws=dict(lw=0.5, ls="--"),
        line_kws_handler=lambda _: dict(lw=3, ls="-"),
        label_kws_handler=lambda _: dict(size=14, weight="bold"),
    )

    fig = circos.plotfig()
    fig.tight_layout()
    circos.ax.legend(loc="upper left", ncol=1, fontsize=14, bbox_to_anchor=(0.7, 1.05))
    fig.savefig("radar.pdf")


def extract_results_to_excel(runs_dir):
    results_list = extract_results(runs_dir)
    player_sides = ["blue", "red"]
    players_metrics = {}
    for match, results in results_list.items():
        if "o1-mini" in match:
            continue
        players = match.split(" vs ")
        if players[0] == players[1]:
            continue
        for player, side in zip(players, player_sides):
            for round_index, round_result in results.items():
                if "round" not in round_index:
                    continue
                metrics = round_result["metrics"][side]
                game_time = round_result["game_time"]
                if player not in players_metrics:
                    players_metrics[player] = {}
                if match not in players_metrics[player]:
                    players_metrics[player][match] = {}
                if round_index not in players_metrics[player][match]:
                    players_metrics[player][match][round_index] = {}
                players_metrics[player][match][round_index]["unit_production"] = (
                    metrics[UNIT_PRODUCTION]
                )
                players_metrics[player][match][round_index]["unit_kills"] = metrics[
                    UNIT_KILLS
                ]
                players_metrics[player][match][round_index]["unit_losses"] = metrics[
                    UNIT_LOSSES
                ]
                players_metrics[player][match][round_index]["damage_dealt"] = metrics[
                    DAMAGE_DEALT
                ]
                players_metrics[player][match][round_index]["damage_taken"] = metrics[
                    DAMAGE_TAKEN
                ]
                players_metrics[player][match][round_index]["resources_spent"] = (
                    metrics[RESOURCES_SPENT]
                )
                players_metrics[player][match][round_index]["resources_harvested"] = (
                    metrics[RESOURCES_HARVESTED]
                )
                players_metrics[player][match][round_index]["game_time"] = game_time

    with pd.ExcelWriter("players_metrics.xlsx", engine="xlsxwriter") as writer:
        for player, matches in players_metrics.items():
            rows = []
            for match, rounds in matches.items():
                for round_index, metrics in rounds.items():
                    row = {"match": match, "round_index": round_index, **metrics}
                    rows.append(row)
            df = pd.DataFrame(rows)

            df.to_excel(writer, sheet_name=player, index=False)


if __name__ == "__main__":
    plot_metrics("runs_logs/runs_llm_vs_llm_8x8basesWorkers")
    # extract_results("PLAP/runs_rule_vs_rule", "rule_vs_rule_16x16_results.log")
    # extract_results("PLAP/runs_Qwen2-72B-Instruct_vs_rule", "qwen_16x16_results.log")
    # extract_results("PLAP/runs_gpt-4o_vs_rule", "gpt-4o_8x8_results.log")
    # extract_results("PLAP/runs_gpt-4o-mini_vs_rule", "gpt-4o-mini_8x8_results.log")
    # extract_results("PLAP/runs_deepseek-chat_vs_rule", "deepseek-chat_8x8_results.log")
    # extract_results("PLAP/runs_llm_vs_llm", "llm_vs_llm_8x8_results.log")
