import numpy as np

from PLAP.utils.utils import path_planning, get_direction

__all__ = [
    "TASK_DEPLOY_UNIT",
    "TASK_HARVEST_MINERAL",
    "TASK_BUILD_BUILDING",
    "TASK_PRODUCE_UNIT",
    "TASK_ATTACK_ENEMY",
    "TASK_SPACE",
    "TASK_ACTION_MAPPING",
]


TASK_DEPLOY_UNIT = "[Deploy Unit]"
TASK_HARVEST_MINERAL = "[Harvest Mineral]"
TASK_BUILD_BUILDING = "[Build Building]"
TASK_PRODUCE_UNIT = "[Produce Unit]"
TASK_ATTACK_ENEMY = "[Attack Enemy]"

TASK_SPACE = [
    TASK_DEPLOY_UNIT,
    TASK_HARVEST_MINERAL,
    TASK_BUILD_BUILDING,
    TASK_PRODUCE_UNIT,
    TASK_ATTACK_ENEMY,
]

DIRECTION_STR_MAPPING = {
    "0": "north",
    "1": "east",
    "2": "south",
    "3": "west",
}


# ====================
#        Tasks
# ====================


def task_deploy_unit(
    unit: dict,
    unit_type: str,
    tgt_loc: tuple,
    path_planner: path_planning,
    action_mask: np.ndarray,
) -> np.ndarray:
    """
    Task for moving to a location

    Args:
        unit (dict): who to do the task
        unit_type (str): type of unit to deploy
        tgt_loc (tuple): where to move
        path_planner (path_planning): to plan the path
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    return move_to_loc(unit, tgt_loc, path_planner, action_mask)


def task_harvest_mineral(
    unit: dict,
    mineral_loc: tuple,
    tgt_loc: tuple,
    base_loc: tuple,
    return_loc: tuple,
    path_planner: path_planning,
    action_mask: np.ndarray,
) -> np.ndarray:
    """
    Task for harvesting minerals

    Args:
        unit (dict): who to do the task
        mineral_loc (tuple): where mineral
        tgt_loc (tuple): where to mine
        base_loc (tuple): where base
        return_loc (tuple) : where to return
        path_planner (path_planning): plans the direction of movement
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    location = tuple(unit["location"])
    if tgt_loc == base_loc == return_loc:
        return noop(unit)
    if unit["resource_num"] == 0 and location != tgt_loc:
        return move_to_loc(unit, tgt_loc, path_planner, action_mask)
    if unit["resource_num"] == 0 and location == tgt_loc:
        return harvest(unit, get_direction(unit["location"], mineral_loc), action_mask)
    if unit["resource_num"] > 0 and location != return_loc:
        return move_to_loc(unit, return_loc, path_planner, action_mask)
    if unit["resource_num"] > 0 and location == return_loc:
        return deliver(unit, get_direction(unit["location"], base_loc), action_mask)


def task_build_building(
    unit: dict,
    building_type: str,
    building_loc: tuple,
    tgt_loc: tuple,
    path_planner: path_planning,
    action_mask: np.ndarray,
) -> np.ndarray:
    """
    Args:
        unit (dict): who to do the task
        building_type (str): what to build
        building_loc (tuple): where building
        tgt_loc (tuple): where to build
        path_planner (path_planning): plans the direction of movement
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    location = tuple(unit["location"])
    if location != tgt_loc:
        return move_to_loc(unit, tgt_loc, path_planner, action_mask)
    return produce(
        unit, get_direction(location, building_loc), building_type, action_mask
    )


def task_produce_unit(
    unit: dict, produce_type: str, direction: str, action_mask: np.ndarray, **kwargs
) -> np.ndarray:
    """
    Task for producing a worker

    Args:
        unit (dict): who to do the task
        produce_type (str): what to produce, ("worker", "light", "heavy" or "ranged")
        direction (str): where to produce
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    return produce(unit, direction, produce_type, action_mask)


def task_attack_enemy(
    unit: dict,
    unit_type: str,
    enemy_type: str,
    enemy_loc: tuple,
    tgt_loc: tuple,
    path_planner: path_planning,
    action_mask: np.ndarray,
) -> np.ndarray:
    """
    Task for attacking a enemy unit

    Args:
        unit (dict): who to do the task
        unit_type (str): what type of unit
        enemy_type (str): what type of enemy
        enemy_loc (tuple): where enemy
        tgt_loc (tuple): where to attack
        path_planner (path_planning): plans the direction of movement
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    location = unit["location"]
    if location != tgt_loc:
        return move_to_loc(unit, tgt_loc, path_planner, action_mask)
    return attack(unit, enemy_loc, action_mask)


# ====================
#      Subtasks
# ====================


def move_to_loc(
    unit: dict, tgt_loc: tuple, path_planner: path_planning, action_mask: np.ndarray
) -> np.ndarray:
    if unit["location"] == tgt_loc:
        return noop(unit)
    direction = path_planner.get_shortest_path(tuple(unit["location"]), tgt_loc)[1]
    if str(direction) in DIRECTION_STR_MAPPING:
        return move(unit, DIRECTION_STR_MAPPING[str(direction)], action_mask)
    return noop(unit)


# ====================
#       Actions
# ====================

ACTION_TYPE_1 = 0
ACTION_TYPE_2 = 6
MOVE_DIRECT_1 = 6
MOVE_DIRECT_2 = 10
HARVEST_DIRECT_1 = 10
HARVEST_DIRECT_2 = 14
RETURN_DIRECT_1 = 14
RETURN_DIRECT_2 = 18
PRODUCE_DIRECT_1 = 18
PRODUCE_DIRECT_2 = 22
PRODUCE_UNIT_1 = 22
PRODUCE_UNIT_2 = 29
ATTACK_PARAM_1 = 29
ATTACK_PARAM_2 = 78

ACTION_INDEX_MAPPING = {
    "noop": 0,
    "move": 1,
    "harvest": 2,
    "return": 3,
    "produce": 4,
    "attack": 5,
}

PRODUCE_UNIT_INDEX_MAPPING = {
    "resource": 0,
    "base": 1,
    "barrack": 2,
    "worker": 3,
    "light": 4,
    "heavy": 5,
    "ranged": 6,
}

DIRECTION_INDEX_MAPPING = {
    "north": 0,
    "east": 1,
    "south": 2,
    "west": 3,
}


def noop(unit: dict) -> np.ndarray:
    """Return a zero vector length of 7."""
    print_action_info(unit, "noop")
    return np.zeros((7), dtype=int)


def move(unit: dict, direction: str, action_mask: np.ndarray) -> np.ndarray:
    """
    Atom action: Move

    Args:
        unit: unit to execute the action
        direction: moving direction
        action_mask: mask for **unit** action types and action parameters, of shape [action types + params]

    Return: action vector, shape of [7]
    """
    if (
        action_mask[ACTION_TYPE_1 + ACTION_INDEX_MAPPING["move"]] == 0
        or action_mask[MOVE_DIRECT_1 + DIRECTION_INDEX_MAPPING[direction]] == 0
    ):
        return noop(unit)
    action = np.zeros((7), dtype=int)
    action[0] = ACTION_INDEX_MAPPING["move"]
    action[1] = DIRECTION_INDEX_MAPPING[direction]
    print_action_info(unit, "move")
    return action


def harvest(unit: dict, direction: str, action_mask: np.ndarray) -> np.ndarray:
    """
    Atom action: Harvest

    Args:
        unit: unit to execute the action
        direction: harvesting direction
        action_mask: mask for **unit** action types and action parameters, of shape [action types + params]

    Return: action vector, shape of [7]
    """
    if (
        action_mask[ACTION_TYPE_1 + ACTION_INDEX_MAPPING["harvest"]] == 0
        or action_mask[HARVEST_DIRECT_1 + DIRECTION_INDEX_MAPPING[direction]] == 0
    ):
        return noop(unit)
    action = np.zeros((7), dtype=int)
    action[0] = ACTION_INDEX_MAPPING["harvest"]
    action[2] = DIRECTION_INDEX_MAPPING[direction]
    print_action_info(unit, "harvest")
    return action


def deliver(unit: dict, direction: str, action_mask: np.ndarray) -> np.ndarray:
    """
    Atom action: Return

    Args:
        unit: unit to execute the action
        direction: delivering direction
        action_mask: mask for **unit** action types and action parameters, of shape [action types + params]

    Return: action vector, shape of [7]
    """
    if (
        action_mask[ACTION_TYPE_1 + ACTION_INDEX_MAPPING["return"]] == 0
        or action_mask[RETURN_DIRECT_1 + DIRECTION_INDEX_MAPPING[direction]] == 0
    ):
        return noop(unit)
    action = np.zeros((7), dtype=int)
    action[0] = ACTION_INDEX_MAPPING["return"]
    action[3] = DIRECTION_INDEX_MAPPING[direction]
    print_action_info(unit, "return")
    return action


def produce(
    unit: dict, direction: str, produce_type: str, action_mask: np.ndarray
) -> np.ndarray:
    """
    Atom action: Produce

    Args:
        unit: unit to execute the action
        direction: production direction
        produce_type: type of unit to produce, 'resource', 'base', 'barrack', 'worker', 'light', 'heavy', 'ranged'
        action_mask: mask for **unit** action types and action parameters, of shape [action types + params]

    Return: action vector, shape of [7]
    """
    if (
        action_mask[ACTION_TYPE_1 + ACTION_INDEX_MAPPING["produce"]] == 0
        or action_mask[PRODUCE_DIRECT_1 + DIRECTION_INDEX_MAPPING[direction]] == 0
        or action_mask[PRODUCE_UNIT_1 + PRODUCE_UNIT_INDEX_MAPPING[produce_type]] == 0
    ):
        return noop(unit)
    action = np.zeros((7), dtype=int)
    action[0] = ACTION_INDEX_MAPPING["produce"]
    action[4] = DIRECTION_INDEX_MAPPING[direction]
    action[5] = PRODUCE_UNIT_INDEX_MAPPING[produce_type]
    print_action_info(unit, f"produce {produce_type}")
    return action


def attack(unit: dict, tgt_loc: tuple, action_mask: np.ndarray) -> np.ndarray:
    """
    Atom action: Attack

    Args:
        unit: unit to execute the action
        tgt_loc: target location for attack
        action_mask: mask for **unit** action types and action parameters, of shape [action types + params]

    Return: action vector, shape of [7]
    """
    unit_loc = tuple(unit["location"])
    try:
        tgt_relative_loc = (tgt_loc[0] - unit_loc[0] + 3) * 7 + (tgt_loc[1] - unit_loc[1] + 3)
    except TypeError as e:
        print(f"{e}\nunit={unit}\ntgt_loc={tgt_loc}")
        return noop(unit)
    if (
        action_mask[ACTION_TYPE_1 + ACTION_INDEX_MAPPING["attack"]] == 0
        or action_mask[ATTACK_PARAM_1 + tgt_relative_loc] == 0
    ):
        return noop(unit)
    action = np.zeros((7), dtype=int)
    action[0] = ACTION_INDEX_MAPPING["attack"]
    action[6] = tgt_relative_loc
    print_action_info(unit, f"attack {tgt_loc}")
    return action


def print_action_info(unit, action):
    if unit["action"] != "noop":
        action = unit["action"]
    print(f"{unit['type']}{tuple(unit['location'])}: {unit['task_type']}{unit['task_params']}/{action}")


# ====================
#     Func. Mapping
# ====================

TASK_ACTION_MAPPING = {
    TASK_DEPLOY_UNIT: task_deploy_unit,
    TASK_HARVEST_MINERAL: task_harvest_mineral,
    TASK_BUILD_BUILDING: task_build_building,
    TASK_PRODUCE_UNIT: task_produce_unit,
    TASK_ATTACK_ENEMY: task_attack_enemy,
}
