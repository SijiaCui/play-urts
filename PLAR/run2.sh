#!/bin/bash

rules_team=("randomBiasedAI" "lightRushAI" "naiveMCTSAI" "workerRushAI" "coacAI")
llm_team=("zero_shot" "few_shot" "zero_shot_w_tips" "few_shot_w_tips")


# Rule vs Rule
for i in $(seq 1 5); do
    for blue in "${rules_team[@]}"; do
        for red in "${rules_team[@]}"; do
            echo "[Rule vs Rule] $blue vs $red (round $i)"
            python aibot_vs_aibot.py --red "$red"  --blue "$blue" --map_index 3
        done
    done
done


# ====================
#         Qwen
# ====================

# # LLM vs Rule
# for i in $(seq 1 5); do
#     for blue in "${llm_team[@]}"; do
#         for red in "${rules_team[@]}"; do
#             echo "[Qwen vs Rule] $blue vs $red (round $i)"
#             python llm_vs_aibot.py --red "$red" --blue_prompt "$blue" --blue "Qwen2-72B-Instruct" --map_index 4
#         done
#     done
# done

# ====================
#      gpt-4o-mini
# ====================

# export http_proxy='http://ccproxy:ccproxy123@10.5.56.27:24789'
# export https_proxy='http://ccproxy:ccproxy123@10.5.56.27:24789'

# export http_proxy='http://ccproxy:ccproxy123@10.7.0.127:7890'
# export https_proxy='http://ccproxy:ccproxy123@10.7.0.127:7890'

# # LLM vs Rule
# for i in $(seq 1 5); do
#     for blue in "${llm_team[@]}"; do
#         for red in "${rules_team[@]}"; do
#             echo "[gpt-4o-mini vs Rule] $blue vs $red (round $i)"
#             python llm_vs_aibot.py --red "$red" --blue_prompt "$blue" --blue "gpt-4o-mini" --max_tokens 256
#         done
#     done
# done

# # ====================
# #        gpt-4o
# # ====================

# # LLM vs Rule
# for i in $(seq 1 5); do
#     for blue in "${llm_team[@]}"; do
#         for red in "${rules_team[@]}"; do
#             echo "[gpt-4o vs Rule] $blue vs $red (round $i)"
#             python llm_vs_aibot.py --red "$red" --blue_prompt "$blue" --blue "gpt-4o" --max_tokens 256
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
#             python llm_vs_aibot.py --red "$red" --blue_prompt "$blue" --blue "deepseek-chat" --max_tokens 256
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