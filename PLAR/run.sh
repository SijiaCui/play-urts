#!/bin/bash

rules_team=("randomBiasedAI" "lightRushAI" "naiveMCTSAI" "workerRushAI" "coacAI")
llm_team=("zero_shot_w_tips" "few_shot_w_tips")


# # Rule vs Rule
# for blue in "${rules_team[@]}"; do
#     for red in "${rules_team[@]}"; do
#         for i in $(seq 1 5); do
#             echo "[Rule vs Rule] $blue vs $red (round $i)"
#             python aibot_vs_aibot.py --red "$red"  --blue "$blue"
#         done
#     done
# done


# ====================
#         Qwen
# ====================

# LLM vs Rule

for i in $(seq 1 5); do
    for blue in "${llm_team[@]}"; do
        for red in "${rules_team[@]}"; do
            echo "[Qwen vs Rule] $blue vs $red (round $i)"
            python llm_vs_aibot.py --red "$red" --blue_prompt "$blue" --blue "Qwen2-72B-Instruct"
        done
    done
done

# ====================
#      gpt-4o-mini
# ====================

# LLM vs Rule
# for i in $(seq 1 5); do
#     for blue in "${llm_team[@]}"; do
#         for red in "${rules_team[@]}"; do
#             echo "[gpt-4o-mini vs Rule] $blue vs $red (round $i)"
#             python llm_vs_aibot.py --red "$red" --blue_prompt "$blue" --blue "gpt-4o-mini"
#         done
#     done
# done

# ====================
#        gpt-4o
# ====================

# LLM vs Rule
# for i in $(seq 1 5); do
#     for blue in "${llm_team[@]}"; do
#         for red in "${rules_team[@]}"; do
#             echo "[gpt-4o vs Rule] $blue vs $red (round $i)"
#             python llm_vs_aibot.py --red "$red" --blue_prompt "$blue" --blue "gpt-4o"
#         done
#     done
# done

# # ====================
# #      DeepSeek
# # ====================

# # LLM vs Rule
# for i in $(seq 1 5); do
#     for blue in "${llm_team[@]}"; do
#         for red in "${rules_team[@]}"; do
#             echo "[DeepSeek vs Rule] $blue vs $red (round $i)"
#             python llm_vs_aibot.py --red "$red" --blue_prompt "$blue" --blue "deepseek-chat"
#         done
#     done
# done

# ====================
#     LLM vs LLM
# ====================
# llms=("Qwen2-72B-Instruct" "deepseek-chat" "gpt-4o-mini" "gpt-4o")

# for i in $(seq 1 5); do
#     for blue in "${llms[@]}"; do
#         for red in "${llms[@]}"; do
#             echo "[LLM vs LLM] $blue vs $red (round $i)"
#             python llm_vs_llm.py --red "$red" --blue "$blue" --blue_prompt "zero_shot" --red_prompt "zero_shot"
#         done
#     done
# done


# ====================
#   Method vs Method
# ====================
# Method vs Method
# for i in $(seq 1 5); do
#     for blue in "${llm_team[@]}"; do
#         for red in "${llm_team[@]}"; do
#             echo "[LLM vs LLM] $blue vs $red (round $i)"
#             python llm_vs_llm.py --red "Qwen2-72B-Instruct" --red_prompt "$red" --blue "Qwen2-72B-Instruct" --blue_prompt "$blue"
#         done
#     done
# done