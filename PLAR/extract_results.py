import os
import numpy as np

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
            'metrics': np.array([...])
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
                metrics.append(list(result.values())[0]["metrics"])
        results_list[match_name]["metrics"] = (
            np.average(metrics, axis=0) if metrics else None
        )
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
        metrics = filter(None, detail.split("Blue")[-1].split("\n")[0].split(" "))
        metrics = [int(x) for x in metrics]
        game_time = int(detail.split("Game over at ")[-1].split(" step")[0])
        point = None
        if winner == "draw":
            point = 0
        else:
            blue, red = run_dir.split("/")[-2].split("_vs_")
            if "blue" in winner or blue == winner:
                point = 1
            elif "red" in winner or red == winner:
                point = -1
        result[f"round {round_index}"]["game_time"] = game_time
        result[f"round {round_index}"]["point"] = point
        result[f"round {round_index}"]["metrics"] = metrics
    return result


def plot_metrics(runs_dir):
    from pycirclize import Circos
    import pandas as pd

    player_metrics = get_avg_metrics(runs_dir)
    data = {NORM_MODEL_NAME_MAP[k]: v for k, v in sorted(player_metrics.items())}

    df = pd.DataFrame.from_dict(data, orient="index")
    df.columns = [
        "Unit Production",
        "Unit Kills",
        "Unit Losses",
        "Damage Dealt",
        "Damage Taken",
        "Resources Spent",
        "Resources Harvested",
    ]
    print(df)

    # Initialize Circos instance for radar chart plot
    circos = Circos.radar_chart(
        df,
        vmax=df.max().max(),
        grid_interval_ratio=0.2,
        show_grid_label=False,
        bg_color=None,
        grid_line_kws=dict(lw=0.5, ls="--"),
        line_kws_handler=lambda _: dict(lw=2, ls="-"),
        label_kws_handler=lambda _: dict(size=10),
    )

    fig = circos.plotfig()
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4)
    fig.savefig("radar.pdf")


def get_avg_metrics(runs_dir: str) -> dict:
    player_metrics = {}
    results_list = extract_results(runs_dir)
    for match, results in results_list.items():
        if "o1-mini" in match:
            continue
        player = match.split(" vs ")[0]
        if player not in player_metrics:
            player_metrics[player] = []
        player_metrics[player].append(results["metrics"])
    player_avg_metrics = {}
    for player, metrics in player_metrics.items():
        player_avg_metrics[player] = np.average(metrics, axis=0)
    return player_avg_metrics


if __name__ == "__main__":
    plot_metrics("backups/backup_runs_llm_vs_llm_8x8basesWorkers")
    # extract_results("PLAR/runs_rule_vs_rule", "rule_vs_rule_16x16_results.log")
    # extract_results("PLAR/runs_Qwen2-72B-Instruct_vs_rule", "qwen_16x16_results.log")
    # extract_results("PLAR/runs_gpt-4o_vs_rule", "gpt-4o_8x8_results.log")
    # extract_results("PLAR/runs_gpt-4o-mini_vs_rule", "gpt-4o-mini_8x8_results.log")
    # extract_results("PLAR/runs_deepseek-chat_vs_rule", "deepseek-chat_8x8_results.log")
    # extract_results("PLAR/runs_llm_vs_llm", "llm_vs_llm_8x8_results.log")
