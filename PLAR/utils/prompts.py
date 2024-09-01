from langchain.prompts import PromptTemplate

'''
提供关键信息如下：

游戏规则：游戏机制，其他有用信息
对手信息：对手的策略和战术，从过去的对局中总结
任务描述：xxx
COA空间：
示例
'''

COA_INSTRUCTION = """You are an expert at playing RTS games and you aim to make a plan to beat the opponent based on all information provided in the following.

## Game Rules
MicroRTS is a classical Real-Time Strategy game, and the following is the manual, including rules and other useful information to master the game. {manual}

## Opponent Information
From the past matches with the opponent, we have summarized the opponent's strategy and tactics as follows. {opponent}

## Task Description
Given the following description of the battlefield situation, your task is to make a deliberate plan to win the fight, specifically, you need to formulate strategies in the form of a Course of Action (COA). Your COA should be in the space of {action_space}.

There are some examples.
{examples}
The Situation: {observation}
Course of Action:
"""


COA_DETAILED_INSTRUCTION = """You are an expert at playing RTS games and you aim to make a plan to beat the opponent based on all information provided in the following.

## Game Rules
MicroRTS is a classical Real-Time Strategy game, and the following is the manual, including rules and other useful information to master the game. {manual}

## Opponent Information
From the past matches with the opponent, we have summarized the opponent's strategy and tactics as follows. {opponent}

## Task Description
Given the following description of the battlefield situation, your task is to make a deliberate plan to win the fight, specifically, you need to formulate your output to include <Situation Analysis>, <Suggested Strategy>, and <Course of Action>. Your COA should be in the space of {action_space}.

There are some examples.
{examples}
The Situation: {observation}
Now, please provide your plan.
"""

''' # maybe useful information
The role of buildings is generally to enable the production of units
The SCV builds structures and harvests minerals and vespene gas.
produce workers (SCVs and Probes respectively) and fighting units from different structures (Command Center and Nexus respectively)

'''

coa_prompt = PromptTemplate(
    input_variables = ["manual", "opponent", "action_space", "examples", "observation"],
    template = COA_INSTRUCTION,
)
