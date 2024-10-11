import os


def extract_results(runs_dir, filename):
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
        match_name = match.replace("_vs_", " vs ")
        results_list[match_name] =  {}
        point = 0
        for run in os.listdir(os.path.join(runs_dir, match)):
            result = get_result(os.path.join(runs_dir, match, run))
            if result:
                results_list[match_name].update(result)
                point += list(result.values())[0]["point"] if list(result.values())[0]["point"] is not None else 0
                results_list[match_name]["point"] = point
    with open(filename, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 32 + "Matches Overview\n")
        f.write("=" * 80 + "\n\n")
        for match, results in sorted(results_list.items()):
            f.write(f"{match} -> {results.get('point', '')}\n")
            for round_index, result in sorted(results.items()):
                if round_index == "point":
                    continue
                f.write(f"{round_index} winner: {result['winner']}\n")
            f.write("\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write(" " * 32 + "Matches Details\n")
        f.write("=" * 80 + "\n\n")
        for match, results in sorted(results_list.items()):
            f.write(match + "\n")
            for round_index, result in sorted(results.items()):
                if round_index == "point":
                    continue
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
        winner = detail.split("The winner is ")[-1].split("\n")[0] if "The winner is" in detail else "draw"
        result[f"round {round_index}"] = {"winner": winner, "detail": detail}
        metrics = filter(None, detail.split("Blue")[-1].split("\n")[0].split(" "))
        metrics = [int(x) for x in metrics]
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
        result[f"round {round_index}"]["metrics"] = metrics
    return result


def plot_metrics():
    ...


if __name__ == "__main__":
    # extract_results("PLAR/runs_rule_vs_rule", "rule_vs_rule_16x16_results.log")
    extract_results("PLAR/runs_Qwen2-72B-Instruct_vs_rule", "qwen_16x16_results.log")
    # extract_results("PLAR/runs_gpt-4o_vs_rule", "gpt-4o_8x8_results.log")
    # extract_results("PLAR/runs_gpt-4o-mini_vs_rule", "gpt-4o-mini_8x8_results.log")
    # extract_results("PLAR/runs_deepseek-chat_vs_rule", "deepseek-chat_8x8_results.log")
    # extract_results("PLAR/runs_llm_vs_llm", "llm_vs_llm_8x8_results.log")
