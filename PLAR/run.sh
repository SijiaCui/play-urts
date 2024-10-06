#!/bin/bash

rules_team=("randomBiasedAI" "naiveMCTSAI" "workerRushAI" "lightRushAI" "coacAI")
llm_team=("zero_shot_prompt" "few_shot_prompt" "prompt_w_tips" "prompt_w_expert_tips")

# LLM vs Rule
for blue in "${llm_team[@]}"; do
    for red in "${rules_team[@]}"; do
        for i in $(seq 1 5); do
            echo "[LLM vs Rule] $blue vs $red (round $i)"
            python llm_vs_aibot.py --red "$red" --blue_prompt "vanilla" "$blue" --blue "Qwen2-72B-Instruct"
        done
    done
done

# Rule vs LLM
for blue in "${rules_team[@]}"; do
    for red in "${llm_team[@]}"; do
        for i in $(seq 1 5); do
            echo "[Rule vs LLM] $blue vs $red (round $i)"
            python llm_vs_aibot.py --blue "$blue" --red_prompt "vanilla" "$red" --blue "Qwen2-72B-Instruct"
        done
    done
done

# LLM vs LLM
for blue in "${llm_team[@]}"; do
    for red in "${llm_team[@]}"; do
        for i in $(seq 1 5); do
            echo "[LLM vs LLM] $blue vs $red (round $i)"
            python llm_vs_llm.py --red "Qwen2-72B-Instruct" --red_prompt "vanilla" "$red" --blue "Qwen2-72B-Instruct" --blue_prompt "vanilla" "$blue"
        done
    done
done

# Rule vs Rule
for blue in "${rules_team[@]}"; do
    for red in "${rules_team[@]}"; do
        for i in $(seq 1 5); do
            echo "[Rule vs Rule] $blue vs $red (round $i)"
            python aibot_vs_aibot.py --red "$red"  --blue "$blue"
        done
    done
done


# LLM (interval 50) vs Rules
for blue in "${llm_team[@]}"; do
    for red in "${rules_team[@]}"; do
        for i in $(seq 1 5); do
            echo "[LLM-interval-50 vs Rules] $blue vs $red (round $i)"
            python llm_vs_aibot.py --red "$red" --blue_prompt "vanilla" "$blue" --blue "Qwen2-72B-Instruct" --tasks_update_interval 50
        done
    done
done