import numpy as np

from typing import List, Dict, Union

from PLAR.task2actions import (
    TASK_SCOUT_LOCATION,
    TASK_HARVEST_MINERAL,
    TASK_BUILD_BUILDING,
    TASK_PRODUCE_UNIT,
    TASK_ATTACK_ENEMY,
    TASK_JOINT_ATTACK_ENEMY,
    TASK_SPACE,
    TASK_ACTION_MAPPING
)
from PLAR.utils.utils import path_planning, manhattan_distance, ENEMY, UNIT_DAMAGE_MAPPING, FIGHT_FOR, UNIT_RANGE_MAPPING

from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder

__all__ = ["script_mapping"]


# ====================
#     Assign Tasks
# ====================


def task_assign(
    task_list: List[str], 
    task_params_list: List, 
    obs_dict: Dict,
    path_planner: path_planning
) -> List[Dict]:
    """Assign tasks to units based on their parameters.

    Args:
        task_list (List[str]): A list of task names.
        task_params (List): A list of task parameters.
        obs_dict (Dict): A dictionary of observations.

    Returns:
        A list of assigned tasks units.
    """
    controlled_units = obs_dict[FIGHT_FOR]
    for task, task_params in zip(task_list, task_params_list):
        if task in ASSIGN_TASK_MAPPING.keys():
            controlled_units = ASSIGN_TASK_MAPPING[task](task_params, controlled_units, obs_dict, path_planner)
    
    # Auto Attack
    assigned_units = []
    for units in controlled_units.values():
        for unit in units:
            if unit["type"] in UNIT_RANGE_MAPPING and unit["action"] == "noop":
                target = auto_choose_target(unit, obs_dict, path_planner)
                if target:
                    unit["task"] = TASK_ATTACK_ENEMY
                    unit["task_params"] = (unit["type"], target["type"], target["location"], unit["location"])
            assigned_units.append(unit)
    return assigned_units

def assign_task_scout_location(task_params: List, controlled_units: Dict, *args) -> Union[Dict, bool]:
    # [Scout Location](unit_type, tgt_loc)
    assigned = False
    for i, unit in enumerate(controlled_units[task_params[0]]):
        if unit["task"] == "[noop]":
            unit["task"] = TASK_SCOUT_LOCATION
            unit["task_params"] = task_params[1:]
            controlled_units[task_params[0]][i] = unit
            assigned = True
            break
    if not assigned:
        print(f"No units found for task {TASK_SCOUT_LOCATION} with params {task_params}")
    return controlled_units


def assign_task_harvest_mineral(
    task_params: List, 
    controlled_units: Dict, 
    obs_dict: Dict, 
    path_planner: path_planning
) -> Union[Dict, bool]:
    # [Harvest Mineral](mineral_loc, tgt_loc, base_loc, return_loc)
    assigned = False
    min_index = 0
    min_path_len = 1e9
    for i, unit in enumerate(controlled_units["worker"]):
        if unit["task"] == "[noop]":
            path1_len, _ = path_planner.get_shortest_path(unit["location"], task_params[1])
            path2_len, _ = path_planner.get_shortest_path(unit["location"], task_params[3])
            if path1_len + path2_len < min_path_len:
                min_index = i
                min_path_len = path1_len + path2_len
    if min_path_len != 1e9:
        controlled_units["worker"][min_index]["task"] = TASK_HARVEST_MINERAL
        controlled_units["worker"][min_index]["task_params"] = task_params
        assigned = True
    if not assigned:
        print(f"No units found for task {TASK_HARVEST_MINERAL} with params {task_params}")
    return controlled_units


def assign_task_build_building(task_params: List, controlled_units: Dict, *args) -> Union[Dict, bool]:
    # [Build Building](building_type, building_loc, tgt_loc)
    assigned = False
    for i, unit in enumerate(controlled_units["worker"]):
        if unit["task"] == "[noop]":
            unit["task"] = TASK_BUILD_BUILDING
            unit["task_params"] = task_params
            controlled_units["worker"][i] = unit
            assigned = True
            break
    if not assigned:
        print(f"No units found for task {TASK_BUILD_BUILDING} with params {task_params}")
    return controlled_units


def assign_task_produce_unit(task_params: List, controlled_units: Dict, *args) -> Union[Dict, bool]:
    # [Produce Unit](produce_type, direction)
    assigned = False
    unit_type = "base" if task_params[0] == "worker" else "barrack"
    for i, unit in enumerate(controlled_units[unit_type]):
        if unit["task"] == "[noop]":
            unit["task"] = TASK_PRODUCE_UNIT
            unit["task_params"] = task_params
            controlled_units[unit_type][i] = unit
            assigned = True
            break
    if not assigned:
        print(f"No units found for task {TASK_PRODUCE_UNIT} with params {task_params}")
    return controlled_units


def assign_task_attack_enemy(
    task_params: List,
    controlled_units: Dict,
    obs_dict: Dict,
    path_planner: path_planning
) -> Union[Dict, bool]:
    # [Attack Enemy](unit_type, enemy_loc, tgt_loc)
    assigned = False
    for i, unit in enumerate(controlled_units[task_params[0]]):
        if unit["task"] == "[noop]":
            task_params = adjust_task_attack_enemy_params(unit, task_params, obs_dict, path_planner)
            if task_params is None:
                return controlled_units
            unit["task"] = TASK_ATTACK_ENEMY if task_params is not None else "[noop]"
            unit["task_params"] = task_params
            controlled_units[task_params[0]][i] = unit
            assigned = True
            break
    if not assigned:
        print(f"No units found for task {TASK_ATTACK_ENEMY} with params {task_params}")
    return controlled_units


def assign_task_joint_attack_enemy(task_params: List, controlled_units: Dict, *args) -> Union[Dict, bool]:
    # TODO: 同时开始攻击
    # [Joint Attack Enemy](units, enemy_loc)
    ...


def adjust_task_attack_enemy_params(
    unit: dict, 
    task_params: List, 
    obs_dict: Dict, 
    path_planner: path_planning
) -> tuple:
    """Adjust attack params based on observation.
    If the enemy leaves the predetermined position, automatically search for the target.

    Args:
        unit (dict): unit to do the task
        task_params (List): attack task params
        obs_dict (Dict): current observation
        path_planner (path_planning): plans path for units

    Returns:
        Tuple: adjusted attack task params
    """
    # [Attack Enemy](unit_type, enemy_type, enemy_loc, tgt_loc)
    unit_type, enemy_type, enemy_loc, tgt_loc = task_params
    enemy = obs_dict["units"][enemy_loc]

    enemy = {} if enemy != {} and enemy["owner"] != ENEMY else enemy
    if enemy == {} or enemy["type"] != enemy_type:  # enemy leave the scheduled location
        enemy_locs = []
        for enemy in obs_dict[ENEMY][enemy_type]:
            enemy_locs.append(enemy["location"])
        if enemy_locs == []:  # no enemy found
            return None
        enemy_loc = enemy_locs[path_planner.get_manhattan_nearest(enemy_loc, enemy_locs)]
    if (
        obs_dict["units"][tgt_loc] != {}
        or manhattan_distance(enemy_loc, tgt_loc) > UNIT_RANGE_MAPPING[unit_type]
    ):  # tgt_loc unusable or enemy out of range
        tgt_locs = path_planner.get_locs_with_dist_from_tgt(
            enemy_loc, UNIT_RANGE_MAPPING[unit_type]
        )
        tgt_loc = tgt_locs[path_planner.get_path_nearest(unit["location"], tgt_locs)]
    return (unit_type, enemy_type, enemy_loc, tgt_loc)


ASSIGN_TASK_MAPPING = {
    TASK_SCOUT_LOCATION: assign_task_scout_location,
    TASK_HARVEST_MINERAL: assign_task_harvest_mineral,
    TASK_BUILD_BUILDING: assign_task_build_building,
    TASK_PRODUCE_UNIT: assign_task_produce_unit,
    TASK_ATTACK_ENEMY: assign_task_attack_enemy,
    TASK_JOINT_ATTACK_ENEMY: assign_task_joint_attack_enemy
}


# ====================
#    Script Mapping
# ====================

def script_mapping(
    env: Union[MicroRTSGridModeVecEnv, VecVideoRecorder], 
    task_list: List[str], 
    task_params: List,
    obs_dict: Dict
) -> np.ndarray:
    """
    Mapping tasks to action vectors.

    Args:
        env (Union[MicroRTSGridModeVecEnv, VecVideoRecorder]): _description_
        obs_dict (Dict): _description_

    Returns:
        np.ndarray: _description_
    """
    height = obs_dict["env"]["height"]
    width = obs_dict["env"]["width"]

    action_masks = env.get_action_mask()
    action_masks = action_masks.reshape(-1, action_masks.shape[-1])

    path_planer = path_planning(compute_valid_map(obs_dict))

    assigned_units = task_assign(task_list, task_params, obs_dict, path_planer)

    action_vectors = np.zeros((height * width, 7), dtype=int)
    for unit in assigned_units:
        if unit["task"] in TASK_SPACE:
            index = unit["location"][0] * width + unit["location"][1]
            action_vectors[index] = TASK_ACTION_MAPPING[unit["task"]](
                unit,
                *unit["task_params"],
                path_planner=path_planer,
                action_mask=action_masks[index],
            )
    return action_vectors


def compute_valid_map(obs_dict):
    valid_map = np.ones((obs_dict["env"]["height"], obs_dict["env"]["width"]), dtype=int)
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
    # 优先攻击可以杀死的重轻远工兵营基地，否则乱打
    loc = unit["location"]
    targets_locs = path_planner.get_locs_with_dist_from_tgt(loc, UNIT_RANGE_MAPPING[unit["type"]])

    targets = [obs_dict["units"][loc] for loc in targets_locs if loc in obs_dict["units"] and obs_dict["units"][loc] != {} and obs_dict["units"][loc]["owner"] == ENEMY]
    
    enemy_hp = np.array([target["hp"] for target in targets if target != {} and target["owner"] == ENEMY])
    enemy_type = np.array([target["type"] for target in targets if target != {} and target["owner"] == ENEMY])

    if len(enemy_hp) == 0:
        return {}
    unit_damage = UNIT_DAMAGE_MAPPING[unit["type"]]
    indices = np.where(enemy_hp <= unit_damage)[0]
    priority_list = ["heavy", "light", "ranged", "worker", "barrack", "base"]

    if len(indices) > 0:
        for target_type in priority_list:
            type_indices = np.where(enemy_type[indices] == target_type)[0]
            if type_indices.size > 0:
                return targets[indices[type_indices[0]]]
    return targets[np.random.choice(len(targets))]


if __name__ == "__main__":
    k = "ranged"
    if k in UNIT_RANGE_MAPPING.keys():
        print(UNIT_RANGE_MAPPING[k])
    else:
        print("Not found")