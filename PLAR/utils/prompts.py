from langchain.prompts import PromptTemplate


# ====================
#   Prompt Templates
# ====================
ZERO_SHOT_PROMPT = """You are an RTS game expert tasked with planning a winning task plan in MicroRTS based on the provided scenario.

## Game Manual
Here are the core rules and mechanics you need to follow: 
{manual}

## Task Space
Develop a winning task plan using allowed task space: 
{task_space}

## Battlefield Situation
Here is the description of the current battlefield: 
{observation}

You are now the {fight_for} side. Please make a task plan to win the game.
"""

PROMPT_W_OPPONENT = """You are an RTS game expert tasked with planning a winning task plan in MicroRTS based on the provided scenario.

## Game Manual
Here are the core rules and mechanics you need to follow: 
{manual}

## Opponent Profile
Based on previous encounters, here is a summary of the opponent strategy and tactics: 
{opponent}

## Task Space
Develop a winning task plan using allowed task space: 
{task_space}

You can refer to the following examples for guidance: 
{examples}

## Battlefield Situation
Here is the description of the current battlefield: 
{observation}

You are now the {fight_for} side. Please make a task plan to win the game.
"""

FEW_SHOT_PROMPT = """You are an RTS game expert tasked with planning a winning task plan in MicroRTS based on the provided scenario.

## Game Manual
Here are the core rules and mechanics you need to follow: 
{manual}

## Task Space
Develop a winning task plan using allowed task space: 
{task_space}

You can refer to the following examples for guidance: 
{examples}

## Battlefield Situation
Here is the description of the current battlefield: 
{observation}

You are now the {fight_for} side. Please make a task plan to win the game.
"""

PROMPT_W_TIP = """You are an RTS game expert tasked with planning a winning task plan in MicroRTS based on the provided scenario.

## Game Manual
Here are the core rules and mechanics you need to follow: 
{manual}

## Task Space
Develop a winning task plan using allowed task space: 
{task_space}

You can refer to the following examples for guidance: 
{examples}

## Battlefield Situation
Here is the description of the current battlefield: 
{observation}

## Tips
Here are some useful tips for you to consider: 
{tip}

You are now the {fight_for} side. Please make a task plan to win the game.
"""

PROMPT_W_OPPONENT = """You are an RTS game expert tasked with planning a winning task plan in MicroRTS based on the provided scenario.

## Game Manual
Here are the core rules and mechanics you need to follow: 
{manual}

## Opponent Profile
Based on previous encounters, here is a summary of the opponent strategy and tactics: 
{opponent}

## Task Space
Develop a winning task plan using allowed task space: 
{task_space}

You can refer to the following examples for guidance: 
{examples}

## Battlefield Situation
Here is the description of the current battlefield: 
{observation}

## Tips
Here are some useful tips for you to consider: 
{tip}

You are now the {fight_for} side. Please make a task plan to win the game.
"""


zero_shot_prompt = PromptTemplate(
    input_variables=["manual", "task_space", "observation", "fight_for"],
    template=ZERO_SHOT_PROMPT,
)

few_shot_prompt = PromptTemplate(
    input_variables=["manual", "task_space", "examples", "observation", "fight_for"],
    template=FEW_SHOT_PROMPT,
)

prompt_w_tip = PromptTemplate(
    input_variables=["manual", "task_space", "examples", "observation", "tip", "fight_for"],
    template=PROMPT_W_TIP,
)

prompt_w_opponent = PromptTemplate(
    input_variables=[
        "manual",
        "opponent",
        "task_space",
        "examples",
        "observation",
        "tip",
        "fight_for"
    ],
    template=PROMPT_W_OPPONENT,
)


# ====================
#      Variables
# ====================
MANUAL = """This is a 2-player grid-based game where all units occupy 1x1 tiles. Each player controls units and can create more by spending a single resource type, which acts as money.

Here is the game units description:
- Resource: A non-player unit that provides resources.
- Base: 10 HP, costs 10 resources, and takes 250 time units to build. Can produce Workers.
- Barracks: 4 HP, costs 5 resources, and takes 200 time units to build. Can produce Light, Heavy, or Ranged units.
- Worker: 1 HP, costs 1 resource, takes 50 time units to build. Can move, attack (1 damage), and harvest mineral.
- Light Unit: 4 HP, costs 2 resources, takes 80 time units to build. Can move and attack (2 damage).
- Heavy Unit: 4 HP, costs 2 resources, takes 120 time units to build. Can move and attack (4 damage).
- Ranged Unit: 1 HP, costs 2 resources, takes 100 time units to build. Can move and attack from 3 distance (1 damage, range 3).
"""

OPPONENT = """
"""

TASK_SPACE = """You are only allowed to utilize the specified tasks to devise your strategy. 
Each task comprises a task name (enclosed in square brackets, e.g. "[Harvest Mineral]") and task parameters (enclosed in parentheses, e.g. "(0, 0)"). 
Your game plan should be a compilation of tasks, delineated by "START of PLAN" and "END of PLAN"

Here are the available tasks and their descriptions:
- [Harvest Mineral] (x, y): Harvest resources from the mineral field located at (x, y).
- [Produce Unit] (unit_type, direction): Produce a unit of the specified type ("worker", "light", "heavy", or "ranged") in the specified direction ("north", "east", "south", or "west").
- [Build Building] (building_type, (x, y)): Build a building of the specified type ("base", "barrack") at the specified location (x, y).
- [Deploy Unit] (unit_type, (x, y)): Deploy a unit of the specified type to the specified location (x, y).
- [Attack ENEMY] (unit_type, ENEMY_type): Use a unit of a specified type ("worker", "light", "heavy", or "ranged") to attack an ENEMY unit of a specified type ("worker", "light", "heavy", "ranged", "base", or "barrack").

Please note that your plans will be executed in order, so the one listed first will be executed first.
"""

BLUE_EXAMPLES = """The Situation: 
Available Mineral Fields: 2
- Mineral Fields(0, 0) resource: 20
- Mineral Fields(7, 7) resource: 20

Red's Units:
base: 1
- (6, 5), action: noop
barrack: 0
worker: 1
- (6, 6), action: noop
light: 0
heavy: 0
ranged: 0

Blue's Units:
base: 1
- (1, 2), task: [noop], action: noop
barrack: 0
worker: 1
- (1, 1), task: [noop], action: noop
light: 0
heavy: 0
ranged: 0

Gaming Plan:
START of TASK
[Harvest Mineral](0, 0),
[Harvest Mineral](0, 0),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'south'),
[Produce Unit]('light', 'south'),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'south'),
[Deploy Unit]('ranged', (4, 5)),
[Deploy Unit]('ranged', (5, 6)),
[Deploy Unit]('ranged', (5, 4)),
[Deploy Unit]('ranged', (7, 4)),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('ranged', 'base'),
[Attack ENEMY]('light', 'base'),
END of TASK
"""

RED_EXAMPLES = """The Situation: 
Available Mineral Fields: 2
- Mineral Fields(0, 0) resource: 20
- Mineral Fields(7, 7) resource: 20

Red's Units:
base: 1
- (6, 5), action: noop
barrack: 0
worker: 1
- (6, 6), action: noop
light: 0
heavy: 0
ranged: 0

Blue's Units:
base: 1
- (1, 2), task: [noop], action: noop
barrack: 0
worker: 1
- (1, 1), task: [noop], action: noop
light: 0
heavy: 0
ranged: 0

Gaming Plan:
START of TASK
[Harvest Mineral](7, 7),
[Harvest Mineral](7, 7),
[Produce Unit]('worker', 'north'),
[Produce Unit]('worker', 'west'),
[Produce Unit]('worker', 'north'),
[Produce Unit]('worker', 'west'),
[Produce Unit]('worker', 'north'),
[Produce Unit]('worker', 'west'),
[Produce Unit]('worker', 'north'),
[Produce Unit]('worker', 'west'),
[Produce Unit]('worker', 'north'),
[Produce Unit]('worker', 'west'),
[Produce Unit]('worker', 'north'),
[Produce Unit]('worker', 'west'),
[Deploy Unit]('worker', (2, 3)),
[Deploy Unit]('worker', (2, 5)),
[Produce Unit]('ranged', 'north'),
[Produce Unit]('ranged', 'west'),
[Produce Unit]('light', 'north'),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'west'),
[Deploy Unit]('ranged', (4, 5)),
[Deploy Unit]('ranged', (5, 6)),
[Deploy Unit]('ranged', (5, 4)),
[Deploy Unit]('ranged', (7, 4)),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('worker', 'base'),
[Attack ENEMY]('ranged', 'base'),
[Attack ENEMY]('light', 'base'),
END of TASK
"""

"""
- 根据生产单位任务设置方向参数，例如，如果需要工人去采矿，设置生产方向靠近矿的边；如果需要部署工人的位置在基地的南方，生产方向设置为南方。
- 两个工人采矿效率会更高。
- 前期多生产产工人并部署在靠近敌人的位置，用于防守。但是不要把工人部署在紧邻基地的位置，这会阻挡基地生产工人。
- 兵营最好建在远离敌人的位置，避免被攻击。
- 多使用远程单位，它可以帮你更轻松地赢得游戏。
"""

TIP = """- Set the direction parameters according to the production unit task. For example, if you need workers to go to the mine, set the production direction close to the edge of the mine; if you need to deploy workers in the south of the base, set the production direction to the south.
- Two workers will be more efficient in mining.
- In the early stage, produce more workers and deploy them near by ENEMY for defense. But don't deploy workers close to the base, which will block the base from producing workers.
- Build a barracks when you have enough resources.
- Use more ranged units, which can help you win the game more easily.
"""
