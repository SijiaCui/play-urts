import os


def extract_results(runs_dir):
    results_list = {}
    matches = os.listdir(runs_dir)
    match_types = os.listdir(runs_dir)
    for match_type in match_types:
        matches = os.listdir(os.path.join(runs_dir, match_type))
        for match in matches:
            match_name = match.replace("_vs_", " vs ")
            results_list[match_name] =  {}
            for run in os.listdir(os.path.join(runs_dir, match_type, match)):
                results_list[match_name].update(get_result(os.path.join(runs_dir, match_type, match, run)))
    with open("results.log", "w") as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 32 + "Matches Overview\n")
        f.write("=" * 80 + "\n\n")
        for match, results in results_list.items():
            f.write(match + "\n")
            for round_index, result in results.items():
                f.write(f"{round_index} winner: {result['winner']}\n")
            f.write("\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write(" " * 32 + "Matches Details\n")
        f.write("=" * 80 + "\n\n")
        for match, results in results_list.items():
            f.write(match + "\n")
            for round_index, result in results.items():
                f.write(f"{round_index} detail: \n{result['detail']}\n")


def get_result(run_dir):
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
    return result


if __name__ == "__main__":
    extract_results("PLAR/Qwen2-72B-Instruct_runs")
    # extract_results("PLAR/gpt-4o-mini_runs")
    # extract_results("PLAR/deepseek-chat_runs")
