#!/bin/bash

rules_team=("randomBiasedAI" "lightRushAI" "naiveMCTSAI" "workerRushAI" "coacAI")  # "randomBiasedAI" "lightRushAI" 
llm_team=("zero_shot_prompt" "few_shot_prompt" "few_shot_w_reflect_tips" "zero_shot_w_expert_tips" "few_shot_w_expert_tips")  # "zero_shot_prompt" "few_shot_prompt" 

# ====================
#      ChatGPT
# ====================
# export http_proxy='http://ccproxy:ccproxy123@10.7.0.127:7890'
# export https_proxy='http://ccproxy:ccproxy123@10.7.0.127:7890'

# # LLM vs Rule
# for blue in "${llm_team[@]}"; do
#     for red in "${rules_team[@]}"; do
#         for i in $(seq 1 5); do
#             echo "[LLM vs Rule] $blue vs $red (round $i)"
#             python llm_vs_aibot.py --red "$red" --blue_prompt "vanilla" "$blue" --blue "gpt-4o-mini" --max_tokens 256
#         done
#     done
# done

# # ====================
# #      DeepSeek
# # ====================

# # LLM vs Rule
# for blue in "${llm_team[@]}"; do
#     for red in "${rules_team[@]}"; do
#         for i in $(seq 1 5); do
#             echo "[LLM vs Rule] $blue vs $red (round $i)"
#             python llm_vs_aibot.py --red "$red" --blue_prompt "vanilla" "$blue" --blue "deepseek-chat" --max_tokens 256
#         done
#     done
# done

# LLM vs LLM
for blue in "${llm_team[@]}"; do
    for red in "${llm_team[@]}"; do
        for i in $(seq 1 5); do
            echo "[LLM vs LLM] $blue vs $red (round $i)"
            python llm_vs_llm.py --red "Qwen2-72B-Instruct" --red_prompt "vanilla" "$red" --blue "Qwen2-72B-Instruct" --blue_prompt "vanilla" "$blue"
        done
    done
done