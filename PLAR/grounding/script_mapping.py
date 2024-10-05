import numpy as np

from typing import List, Dict, Union, Tuple

__all__ = ["script_mapping"]

from .task2actions import (
    TASK_DEPLOY_UNIT,
    TASK_HARVEST_MINERAL,
    TASK_BUILD_BUILDING,
    TASK_PRODUCE_UNIT,
    TASK_ATTACK_ENEMY,
    TASK_ACTION_MAPPING,
)
from PLAR.utils.utils import (
    path_planning,
    get_direction,
    UNIT_DAMAGE_MAPPING,
    UNIT_RANGE_MAPPING,
)

import PLAR.utils as utils

from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder

__all__ = ["script_mapping"]


# ====================
#     Assign Tasks
# ====================


def task_assign(
    tasks: List[Tuple],
    obs_dict: Dict,
    path_planner: path_planning
) -> List[Dict]:
    """Assign tasks to units based on their parameters.

    Args:
        tasks (List[Tuple]): list of tasks, each task is a tuple of (task_type, task_params)
        obs_dict (Dict): observations dictionary
        path_planner (path_planning): plans path for units

    Returns:
        List[Dict]: list of assigned units
    """
    exist_units = {
        unit_id: unit_dict
        for unit_id, unit_dict in obs_dict[utils.FIGHT_FOR].items()
        if isinstance(unit_dict, dict)
    }

    for task in tasks:
        exist_units = ASSIGN_TASK_MAPPING[task[0]](task[1], exist_units, obs_dict, path_planner)

    assigned_units = []
    for unit in exist_units.values():
        if unit["type"] in UNIT_RANGE_MAPPING and unit["action"] == "noop":
            # Auto Attack
            target = auto_choose_target(unit, obs_dict, path_planner)
            if target:
                unit["task_type"] = TASK_ATTACK_ENEMY
                unit["task_params"] = (unit["type"], target["type"])
        if unit["task_type"] != "[noop]":
            assigned_units.append(unit)
    return assigned_units


def assign_task_deploy_unit(
    task_params: List,
    exist_units: Dict,
    obs_dict: Dict,
    path_planner: path_planning,
) -> Union[Dict, bool]:
    # [Deploy Unit](unit_type, tgt_loc)
    closest_unit_id = -1
    min_path_len = 1e9
    for unit in exist_units.values():
        if unit["task_type"] == "[noop]" and unit["type"] == task_params[0]:
            path_len, _ = path_planner.get_shortest_path(
                unit["location"], task_params[1]
            )
            if path_len < min_path_len:
                closest_unit_id = unit["id"]
                min_path_len = path_len
    if closest_unit_id != -1:
        exist_units[closest_unit_id]["task_type"] = TASK_DEPLOY_UNIT
        exist_units[closest_unit_id]["task_params"] = task_params
    else:
        print(f"Pending task: {TASK_DEPLOY_UNIT}{task_params}")
    return exist_units


def assign_task_harvest_mineral(
    task_params: List, exist_units: Dict, obs_dict: Dict, path_planner: path_planning
) -> Union[Dict, bool]:
    # [Harvest Mineral](mineral_loc)
    closest_unit_id = -1
    min_path_len = 1e9
    for unit in exist_units.values():
        if unit["task_type"] == "[noop]" and unit["type"] == "worker":
            path_len, _ = path_planner.get_shortest_path(unit["location"], task_params)
            if path_len < min_path_len:
                closest_unit_id = unit["id"]
                min_path_len = path_len
    if closest_unit_id != -1:
        exist_units[closest_unit_id]["task_type"] = TASK_HARVEST_MINERAL
        exist_units[closest_unit_id]["task_params"] = task_params
    else:
        print(f"Pending task: {TASK_HARVEST_MINERAL}{task_params}")
    return exist_units


def assign_task_build_building(
    task_params: List, exist_units: Dict, obs_dict: Dict, path_planner: path_planning
) -> Union[Dict, bool]:
    # [Build Building](building_type, building_loc)
    closest_unit_id = -1
    min_path_len = 1e9
    for unit in exist_units.values():
        if unit["task_type"] == "[noop]" and unit["type"] == "worker":
            path_len, _ = path_planner.get_shortest_path(
                unit["location"], task_params[1]
            )
            if path_len < min_path_len:
                closest_unit_id = unit["id"]
                min_path_len = path_len
    if closest_unit_id != -1:
        exist_units[closest_unit_id]["task_type"] = TASK_BUILD_BUILDING
        exist_units[closest_unit_id]["task_params"] = task_params
    else:
        print(f"Pending task: {TASK_BUILD_BUILDING}{task_params}")
    return exist_units


def assign_task_produce_unit(
    task_params: List,
    exist_units: Dict,
    obs_dict: Dict,
    path_planner: path_planning,
) -> Union[Dict, bool]:
    # [Produce Unit](produce_type, direction)
    assigned = False
    unit_type = "base" if task_params[0] == "worker" else "barrack"
    for unit in exist_units.values():
        if unit["task_type"] == "[noop]" and unit["type"] == unit_type:
            exist_units[unit["id"]]["task_type"] = TASK_PRODUCE_UNIT
            exist_units[unit["id"]]["task_params"] = task_params
            assigned = True
            break
    if not assigned:
        print(f"Pending task: {TASK_PRODUCE_UNIT}{task_params}")
    return exist_units


def assign_task_attack_enemy(
    task_params: List, exist_units: Dict, obs_dict: Dict, path_planner: path_planning
) -> Union[Dict, bool]:
    # [Attack Enemy](unit_type, enemy_type)
    closest_dist = {}
    enemy_locs = []
    for enemy in obs_dict[utils.ENEMY].values():
        if isinstance(enemy, dict) and enemy["type"] == task_params[1]:
            enemy_locs.append(enemy["location"])
    if len(enemy_locs) == 0:
        print(f"Pending task: {TASK_ATTACK_ENEMY}{task_params}")
        return exist_units
    for unit in exist_units.values():
        if unit["task_type"] == "[noop]" and unit["type"] == task_params[0]:
            nearest_loc = path_planner.get_path_nearest(unit["location"], enemy_locs)
            closest_dist[unit["id"]], _ = path_planner.get_shortest_path(
                unit["location"], nearest_loc
            )
    if len(closest_dist) > 0:
        unit_id = min(closest_dist, key=closest_dist.get)
        exist_units[unit_id]["task_type"] = TASK_ATTACK_ENEMY
        exist_units[unit_id]["task_params"] = task_params
        return exist_units
    else:
        print(f"Pending task: {TASK_ATTACK_ENEMY}{task_params}")
        return exist_units


ASSIGN_TASK_MAPPING = {
    TASK_DEPLOY_UNIT: assign_task_deploy_unit,
    TASK_HARVEST_MINERAL: assign_task_harvest_mineral,
    TASK_BUILD_BUILDING: assign_task_build_building,
    TASK_PRODUCE_UNIT: assign_task_produce_unit,
    TASK_ATTACK_ENEMY: assign_task_attack_enemy,
}


# ====================
#    Script Mapping
# ====================


def script_mapping(
    env: Union[MicroRTSGridModeVecEnv, VecVideoRecorder],
    tasks: List[Tuple],
    obs_dict: Dict
) -> np.ndarray:
    """
    Mapping tasks to action vectors.

    Args:
        env (Union[MicroRTSGridModeVecEnv, VecVideoRecorder]): game environment
        tasks (List[Tuple]): list of tasks, each task is a tuple of (task_type, task_params)
        obs_dict (Dict): observation dictionary

    Returns:
        np.ndarray: action vectors
        List[Dict]: updated assigned units
    """
    height = obs_dict["env"]["height"]
    width = obs_dict["env"]["width"]

    action_masks = env.get_action_mask()
    action_masks = action_masks[0] if utils.FIGHT_FOR == "blue" else action_masks[1]
    action_masks = action_masks.reshape(-1, action_masks.shape[-1])

    path_planer = path_planning(compute_valid_map(obs_dict))
    assigned_units = task_assign(tasks, obs_dict, path_planer)

    action_vectors = np.zeros((height * width, 7), dtype=int)
    for unit in assigned_units:
        index = unit["location"][0] * width + unit["location"][1]
        task_params = ADAPT_TASK_PARAMS_MAPPING[unit["task_type"]](unit, obs_dict, path_planer)
        action_vectors[index] = TASK_ACTION_MAPPING[unit["task_type"]](
            unit,
            *task_params,
            path_planner=path_planer,
            action_mask=action_masks[index],
        )
    return action_vectors


def adapt_task_harvest_mineral_params(unit, obs_dict, path_planner: path_planning):
    # (mineral_loc) -> (mineral_loc, tgt_loc, base_loc, return_loc)
    mineral_loc = unit["task_params"]
    tgt_locs = get_around_locs(mineral_loc, obs_dict)
    for loc in tgt_locs:
        if obs_dict["units"][loc] != {} and unit["location"] != loc:
            tgt_locs.remove(loc)
    tgt_loc = path_planner.get_manhattan_nearest(unit["location"], tgt_locs)
    for _unit in obs_dict[utils.FIGHT_FOR].values():
        if isinstance(_unit, dict) and _unit["type"] == "base":
            base_loc = _unit["location"]
            return_locs = get_around_locs(base_loc, obs_dict)
            for loc in return_locs:
                if obs_dict["units"][loc] != {} and unit["location"] != loc:
                    return_locs.remove(loc)
            return_loc = path_planner.get_manhattan_nearest(unit["location"], return_locs)
            return (mineral_loc, tgt_loc, base_loc, return_loc)
    return (mineral_loc, unit["location"], unit["location"], unit["location"])


def adapt_task_build_building_params(unit, obs_dict, path_planner: path_planning):
    # (building_type, building_loc) -> (building_type, building_loc, tgt_loc)
    building_type, building_loc = unit["task_params"]
    tgt_locs = get_around_locs(building_loc, obs_dict)
    for tgt_loc in tgt_locs:
        if obs_dict["units"][tgt_loc] != {} and unit["location"] != tgt_loc:
            tgt_locs.remove(tgt_loc)
    tgt_loc = path_planner.get_path_nearest(unit["location"], tgt_locs)
    return (building_type, building_loc, tgt_loc)


def adapt_task_attack_enemy_params(unit, obs_dict, path_planner: path_planning):
    # [Attack Enemy](unit_type, enemy_type) -> [Attack Enemy](unit_type, enemy_type, enemy_loc, tgt_loc)
    unit_type, enemy_type = unit["task_params"]
    enemy_locs = []
    for enemy in obs_dict[utils.ENEMY].values():
        if isinstance(enemy, dict) and enemy["type"] == enemy_type:
            enemy_locs.append(enemy["location"])
    if len(enemy_locs) == 0:
        return (unit_type, enemy_type, (0, 0), unit["location"])
    enemy_loc = path_planner.get_path_nearest(unit["location"], enemy_locs)

    tgt_locs = path_planner.get_locs_with_dist_to_tgt(
        enemy_loc, UNIT_RANGE_MAPPING[unit_type]
    )
    tgt_loc = path_planner.get_path_nearest(unit["location"], tgt_locs)
    return (unit_type, enemy_type, enemy_loc, tgt_loc)


def adapt_task_deploy_unit_params(unit, obs_dict, path_planner: path_planning):
    return unit["task_params"]


def adapt_task_produce_unit_params(unit, obs_dict, path_planner: path_planning):
    # [Produce Unit](produce_type, direction)
    direction = unit["task_params"][1]
    loc = get_direction_loc(unit, direction)
    if loc in obs_dict["units"] and obs_dict["units"][loc] == {}:
        return unit["task_params"]
    around_locs = get_around_locs(unit["location"], obs_dict)
    for loc in around_locs:
        if loc in obs_dict["units"] and obs_dict["units"][loc] == {}:
            return (unit["task_params"][0], get_direction(unit["location"], loc))
    return unit["task_params"]


ADAPT_TASK_PARAMS_MAPPING = {
    TASK_ATTACK_ENEMY: adapt_task_attack_enemy_params,
    TASK_DEPLOY_UNIT: adapt_task_deploy_unit_params,
    TASK_PRODUCE_UNIT: adapt_task_produce_unit_params,
    TASK_HARVEST_MINERAL: adapt_task_harvest_mineral_params,
    TASK_BUILD_BUILDING: adapt_task_build_building_params,
}


def get_direction_loc(unit, direction):
    if direction == "north":
        return (unit["location"][0] - 1, unit["location"][1])
    elif direction == "south":
        return (unit["location"][0] + 1, unit["location"][1])
    elif direction == "east":
        return (unit["location"][0], unit["location"][1] + 1)
    elif direction == "west":
        return (unit["location"][0], unit["location"][1] - 1)


def get_around_locs(loc, obs_dict):
    around_locs = [
        (loc[0] + 1, loc[1]),
        (loc[0] - 1, loc[1]),
        (loc[0], loc[1] + 1),
        (loc[0], loc[1] - 1),
    ]
    for around_loc in around_locs:
        if around_loc not in obs_dict["units"].keys():
            around_locs.remove(around_loc)
    return around_locs


def compute_valid_map(obs_dict):
    valid_map = np.ones(
        (obs_dict["env"]["height"], obs_dict["env"]["width"]), dtype=int
    )
    for loc, unit in obs_dict["units"].items():
        if unit != {}:
            valid_map[unit["location"][0], unit["location"][1]] = 0
    return valid_map


def auto_choose_target(unit: dict, obs_dict: dict, path_planner: path_planning) -> dict:
    """Auto choice target to attack.
    If there is a target that can be killed, attack it.
    Otherwise, randomly choose a target to attack.

    Args:
        unit (dict): who to do the attack
        obs_dict (dict): observation dict
        path_planner (path_planning): path planner

    Returns: chosen target
    """
    # 优先攻击可以杀死的重轻远工兵营基地
    # 如果都不能一击毙命，则优先攻击重轻远工兵营基地
    loc = unit["location"]
    targets_locs = path_planner.get_locs_with_dist_to_tgt(
        loc, UNIT_RANGE_MAPPING[unit["type"]]
    )

    targets = [
        obs_dict["units"][loc]
        for loc in targets_locs
        if loc in obs_dict["units"]
        and obs_dict["units"][loc] != {}
        and obs_dict["units"][loc]["owner"] == utils.ENEMY
    ]

    enemy_hp = np.array(
        [
            target["hp"]
            for target in targets
            if target != {} and target["owner"] == utils.ENEMY
        ]
    )
    enemy_type = np.array(
        [
            target["type"]
            for target in targets
            if target != {} and target["owner"] == utils.ENEMY
        ]
    )

    if len(enemy_hp) == 0:
        return {}
    unit_damage = UNIT_DAMAGE_MAPPING[unit["type"]]
    indices = np.where(enemy_hp <= unit_damage)[0]
    priority_list = ["heavy", "light", "ranged", "worker", "barrack", "base"]

    indices = indices if len(indices) > 0 else np.arange(len(targets))
    for target_type in priority_list:
        type_indices = np.where(enemy_type[indices] == target_type)[0]
        if type_indices.size > 0:
            return targets[indices[type_indices[0]]]
    return targets[np.random.choice(len(targets))]
