#!/bin/bash

red_team=("naiveMCTSAI" "workerRushAI")
blue_team=("few_shot_prompt")


for blue in "${blue_team[@]}"; do
    for red in "${red_team[@]}"; do
        for i in $(seq 1 5); do
            echo "Running round $i with --red $red and --blue $blue"
            python llm_vs_aibot.py --red "$red" --blue_prompt "vanilla" "$blue"
        done
    done
done

# for blue in "${blue_team[@]}"; do
#     for red in "${blue_team[@]}"; do
#         for i in $(seq 1 5); do
#             echo "Running round $i with --red $red and --blue $blue"
#             python llm_vs_llm.py --red "Qwen2-72B-Instruct" --red_prompt "vanilla" "$red" --blue_prompt "vanilla" "$blue"
#         done
#     done
# done