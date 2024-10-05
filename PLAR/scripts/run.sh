#!/bin/bash

# for i in {1..5}
# do
#   echo "Running iteration $i..."
#   python llm_vs_aibot.py --red randomAI
# done

# for i in {1..5}
# do
#   echo "Running iteration $i..."
#   python llm_vs_aibot.py --red randomBiasedAI
# done

# for i in {1..5}
# do
#   echo "Running iteration $i..."
#   python llm_vs_aibot.py --red naiveMCTSAI
# done

for i in {1..5}
do
  echo "Running iteration $i..."
  python llm_vs_aibot.py --red workerRushAI
done

for i in {1..5}
do
  echo "Running iteration $i..."
  python llm_vs_aibot.py --red lightRushAI
done

for i in {1..5}
do
  echo "Running iteration $i..."
  python llm_vs_aibot.py --red passiveAI
done

for i in {1..5}
do
  echo "Running iteration $i..."
  python llm_vs_aibot.py --red coacAI
done