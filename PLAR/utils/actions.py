import numpy as np

__all__ = ["noop", "move", "harvest", "deliver", "produce", "attack"]

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


def produce(unit: dict, direction: str, produce_type: str, action_mask: np.ndarray) -> np.ndarray:
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
    tgt_relative_loc = (tgt_loc[0] - unit_loc[0] + 3) * 7 + (tgt_loc[1] - unit_loc[1] + 3)
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
    print(f"{unit['type']}{tuple(unit['location'])}: {unit['task']}/{action}")


if __name__ == "__main__":
    action_mask = np.ones((78), dtype=int)
    unit = {
        "owner": "blue",
        "type": "worker",
        "location": [0, 1],
        "hp": 1,
        "resource_num": 1,
        "action": "move",
        "task": "[Harvest Mineral]",
    }
    print(noop(unit))
    print(move(unit, 'west', action_mask))
    print(harvest(unit, 'west', action_mask))
    print(deliver(unit, 'west', action_mask))
    print(produce(unit, 'west', 'worker', action_mask))
    print(attack(unit, (2, 2), action_mask))
