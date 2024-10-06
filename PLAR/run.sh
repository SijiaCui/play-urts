#!/bin/bash

red_team=("randomBiasedAI" "naiveMCTSAI" "workerRushAI" "lightRushAI" "coacAI")
blue_team=("zero_shot_prompt" "few_shot_prompt" "prompt_w_tips")


for blue in "${blue_team[@]}"; do
    for red in "${red_team[@]}"; do
        for i in $(seq 1 5); do
            echo "Running round $i with --red $red and --blue $blue"
            python llm_vs_aibot.py --red "$red" --blue_prompt "vanilla" "$blue"
        done
    done
done

for blue in "${blue_team[@]}"; do
    for red in "${blue_team[@]}"; do
        for i in $(seq 1 5); do
            echo "Running round $i with --red $red and --blue $blue"
            python llm_vs_llm.py --red "Qwen2-72B-Instruct" --red_prompt "vanilla" "$red" --blue_prompt "vanilla" "$blue"
        done
    done
done