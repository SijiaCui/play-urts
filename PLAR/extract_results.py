import os
import yaml


def extract_results(runs_dir):
    results = []
    matches = os.listdir(runs_dir)
    for match in matches:
        for run in os.listdir(os.path.join(runs_dir, match)):
            result = get_results(os.path.join(runs_dir, match, run))
            if result is not None:
                results.append(result)
    matches_statistics = get_matches_statistics(results)
    with open("results.log", "w") as f:
        for match, statistics in matches_statistics.items():
            f.write(match + "\n")
            f.write(str(statistics))
            f.write("\n\n\n")
        for result in results:
            f.write(result["match"] + "\n")
            f.write(result["result"])
            f.write("\n")


def get_results(run_dir):
    result = {}
    with open(os.path.join(run_dir, "configs.yaml"), "r") as f:
        config = yaml.safe_load(f)
    blue = f"{config['blue']}-{config['blue_prompt'][0]}-{config['blue_prompt'][1]}"
    red = f"{config['red']}" if "AI" in config['red'] else f"{config['red']}-{config['red_prompt'][0]}-{config['red_prompt'][1]}"
    result["match"] = f"{blue} vs {red}"
    with open(os.path.join(run_dir, "run.log"), "r") as f:
        text = f.read()
    if "Game over" in text:
        result["result"] = "Game over" + text.split("Game over")[-1]
        return result
    else:
        return None


def get_matches_statistics(results):
    blue_vs_red_statistics = {}
    for result in results:
        if result["match"] not in blue_vs_red_statistics:
            blue_vs_red_statistics[result["match"]] = []
        winner = result["result"].split("The winner is ")[-1].split("\n")[0] if "The winner is" in result["result"] else "Draw"
        blue_vs_red_statistics[result["match"]].append(winner)
    return blue_vs_red_statistics


if __name__ == '__main__':
    results = extract_results("PLAR/runs")
