import numpy as np

from PLAR.utils.actions import noop, move, harvest, deliver, produce, attack
from PLAR.utils.utils import (
    path_planning,
    manhattan_distance,
    get_direction,
    build_place_invalid,
)


TASK_HARVEST_MINERAL = "[Harvest Mineral]"
TASK_BUILD_BASE = "[Build Base]"
TASK_BUILD_BARRACK = "[Build Barrack]"

TASK_PRODUCE_WORKER = "[Produce Worker]"
TASK_PRODUCE_LIGHT = "[Produce Light Soldier]"
TASK_PRODUCE_HEAVY = "[Produce Heavy Soldier]"
TASK_PRODUCE_RANGED = "[Produce Ranged Soldier]"

TASK_ATTACK_WORKER = "[Attack Enemy Worker]"
TASK_ATTACK_BUILDINGS = "[Attack Enemy Buildings]"
TASK_ATTACK_SOLDIERS = "[Attack Enemy Soldiers]"

TASK_MOVE_TO_LOCATION = "[Move to Location]"

TASK_SPACE = [
    TASK_HARVEST_MINERAL,
    TASK_BUILD_BASE,
    TASK_BUILD_BARRACK,
    TASK_PRODUCE_WORKER,
    TASK_PRODUCE_LIGHT,
    TASK_PRODUCE_HEAVY,
    TASK_PRODUCE_RANGED,
    TASK_ATTACK_WORKER,
    TASK_ATTACK_BUILDINGS,
    TASK_ATTACK_SOLDIERS,
    TASK_MOVE_TO_LOCATION,
]

DIRECTION_STR_MAPPING = {"0": "north", "1": "east", "2": "south", "3": "west"}


FIGHT_FOR = "blue"
ENEMY = "red"


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
    if unit["resource_num"] == 0 and location != tgt_loc:
        return move_to_loc(unit, tgt_loc, path_planner, action_mask)
    if unit["resource_num"] == 0 and location == tgt_loc:
        return harvest_mineral(unit, mineral_loc, action_mask)
    if unit["resource_num"] > 0 and location != return_loc:
        return move_to_loc(unit, return_loc, path_planner, action_mask)
    if unit["resource_num"] > 0 and location == return_loc:
        return return_mineral(unit, base_loc, action_mask)


def task_build_base(
    unit: dict,
    base_loc: tuple,
    tgt_loc: tuple,
    path_planner: path_planning,
    action_mask: np.ndarray,
) -> np.ndarray:
    """
    Task for building a base

    Args:
        unit (dict): who to do the task
        base_loc (tuple): where building
        tgt_loc (tuple): where to build
        path_planner (path_planning): plans the direction of movement
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    location = tuple(unit["location"])
    if location != tgt_loc:
        return move_to_loc(unit, tgt_loc, path_planner, action_mask)
    return build(unit, "base", base_loc, action_mask)


def task_build_barrack(
    unit: dict,
    barrack_loc: tuple,
    tgt_loc: tuple,
    path_planner: path_planning,
    action_mask: np.ndarray,
) -> np.ndarray:
    """
    Task for building a barrack

    Args:
        unit (dict): who to do the task
        barrack_loc (tuple): where building
        tgt_loc (tuple): where to build
        path_planner (path_planning): plans the direction of movement
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    location = tuple(unit["location"])
    if location != tgt_loc:
        return move_to_loc(unit, tgt_loc, path_planner, action_mask)
    return build(unit, "barrack", barrack_loc, action_mask)


def task_produce_worker(
    unit: dict, output_loc: tuple, action_mask: np.ndarray
) -> np.ndarray:
    """
    Task for producing a worker

    Args:
        unit (dict): who to do the task
        output_loc (tuple): where to produce
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    return produce(
        unit, get_direction(tuple(unit["location"]), output_loc), "worker", action_mask
    )


def task_produce_light(
    unit: dict, output_loc: tuple, action_mask: np.ndarray
) -> np.ndarray:
    """
    Task for producing a light

    Args:
        unit (dict): who to do the task
        output_loc (tuple): where to produce
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    return produce(
        unit, get_direction(tuple(unit["location"]), output_loc), "light", action_mask
    )


def task_produce_heavy(
    unit: dict, output_loc: tuple, action_mask: np.ndarray
) -> np.ndarray:
    """
    Task for producing a heavy

    Args:
        unit (dict): who to do the task
        output_loc (tuple): where to produce
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    return produce(
        unit, get_direction(tuple(unit["location"]), output_loc), "heavy", action_mask
    )


def task_produce_ranged(
    unit: dict, output_loc: tuple, action_mask: np.ndarray
) -> np.ndarray:
    """
    Task for producing a ranged

    Args:
        unit (dict): who to do the task
        output_loc (tuple): where to produce
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    return produce(
        unit, get_direction(tuple(unit["location"]), output_loc), "ranged", action_mask
    )


def task_attack_worker(
    unit: dict,
    attack_loc: dict,
    tgt_loc: tuple,
    path_planner: path_planning,
    action_mask: np.ndarray,
) -> np.ndarray:
    """
    Task for attacking a worker

    Args:
        unit (dict): who to do the task
        attack_loc (dic)
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    return attack(
        unit, get_direction(tuple(unit["location"]), attack_loc), "worker", action_mask
    )


def task_move_to_loc(
    unit: dict, tgt_loc: tuple, path_planner: path_planning, action_mask: np.ndarray
) -> np.ndarray:
    """
    Task for moving to a location

    Args:
        unit (dict): who to do the task
        tgt_loc (tuple): where to move
        path_planner (path_planning): to plan the path
        action_mask (np.ndarray): to indicate which actions are available, of shape [action types + params]
    """
    return move_to_loc(unit, tgt_loc, path_planner, action_mask)


def adapt_param_task_move_to_loc(
    unit: dict,
    tgt_loc: tuple,
    path_planner: path_planning,
    action_mask: np.ndarray,
    env: dict,
) -> tuple: ...


def attack_unit(
    unit: dict,
    attack_loc: dict,
    tgt_loc: tuple,
    path_planner: path_planning,
    action_mask: np.ndarray,
) -> np.ndarray:
    location = tuple(unit["location"])

    # is under attack range
    assert (
        unit["type"] in ["worker", "light", "heavy"]
        and manhattan_distance(tgt_loc, attack_loc) == 1
    ) or (
        unit["type"] == "ranged" and manhattan_distance(tgt_loc, attack_loc) <= 3
    ), f"{unit['type'].capitalize()}{unit['location']} can't attack {attack_loc} at {tgt_loc}"
    if location != tgt_loc:
        return move_to_loc(unit, tgt_loc, path_planner, action_mask)
    return attack(unit, attack_loc, action_mask)


def move_to_loc(
    unit: dict, tgt_loc: tuple, path_planner: path_planning, action_mask: np.ndarray
) -> np.ndarray:
    direction = path_planner.get_shortest_path(tuple(unit["location"]), tgt_loc)[1]
    return move(unit, DIRECTION_STR_MAPPING[str(direction)], action_mask)


def harvest_mineral(
    unit: dict, mineral_loc: tuple, action_mask: np.ndarray
) -> np.ndarray:
    location = tuple(unit["location"])
    assert (
        manhattan_distance(location, mineral_loc) == 1
    ), f"Can't mine{mineral_loc} at {location}."
    return harvest(unit, get_direction(location, mineral_loc), action_mask)


def return_mineral(unit: dict, base_loc: tuple, action_mask: np.ndarray) -> np.ndarray:
    location = tuple(unit["location"])
    assert (
        manhattan_distance(location, base_loc) == 1
    ), f"Can't return base{base_loc} at {location}."
    return deliver(unit, get_direction(location, base_loc), action_mask)


def build(
    unit: dict, building_type: str, building_loc: tuple, action_mask: np.ndarray
) -> np.ndarray:
    location = tuple(unit["location"])
    assert (
        manhattan_distance(location, building_loc) == 1
    ), f"Can't build {building_type}{building_loc} at {location}."
    return produce(
        unit, get_direction(location, building_loc), building_type, action_mask
    )


TASK_MAPPING = {
    TASK_MOVE_TO_LOCATION: [task_move_to_loc, adapt_param_task_move_to_loc],
    TASK_HARVEST_MINERAL: task_harvest_mineral,  # adapt_param_task_harvest_mineral 当矿位置错了，自动找目标矿附近的矿
    TASK_BUILD_BASE: task_build_base,
    TASK_BUILD_BARRACK: task_build_barrack,
    TASK_PRODUCE_WORKER: task_produce_worker,  # 每个时间步观察，人在不在原来的目标位置，如果已经走了，自动去攻击原攻击对象位置附近的工人
    TASK_PRODUCE_LIGHT: task_produce_light,
    TASK_PRODUCE_HEAVY: task_produce_heavy,
    TASK_PRODUCE_RANGED: task_produce_ranged,
    TASK_ATTACK_WORKER: task_attack_worker,
}


def find_around_enemies(unit, obs_json):
    enemies = obs_json[ENEMY]
    around_locs = [
        (unit["location"][0], unit["location"][1] - 1),  # north
        (unit["location"][0] - 1, unit["location"][1]),  # east
        (unit["location"][0], unit["location"][1] + 1),  # south
        (unit["location"][0] + 1, unit["location"][1]),  # west
    ]
    attack_enemies = []
    for enemy in enemies:
        if tuple(enemy["location"]) in around_locs:
            attack_enemies.append(enemy)
    return attack_enemies


def script_mapping(env, obs, obs_json) -> np.ndarray:
    from PLAR.utils.utils import build_place_invalid

    height = obs_json["env"]["height"]
    width = obs_json["env"]["width"]
    action_masks = env.get_action_mask()
    action_masks = action_masks.reshape(-1, action_masks.shape[-1])

    # generate a valid map that indicates that grid is valid to be moved on
    obs = obs.reshape((height, width, -1))
    valid_map = np.zeros(shape=(height, width))
    valid_map[np.where(obs[:, :, 13] == 1)] = 1  # UNIT_NONE_INDEX
    valid_map = build_place_invalid(obs, valid_map)

    path_planer = path_planning(valid_map)

    actions = np.zeros((len(action_masks), 7), dtype=int)

    for unit in obs_json[FIGHT_FOR]:
        index = unit["location"][0] * width + unit["location"][1]
        task = unit["task"]
        if "aggressive" in unit["task_params"]:
            attack_enemies = find_around_enemies(unit, obs_json)
            if len(attack_enemies) > 0:
                attack_loc = np.random.choice(attack_enemies)[
                    "location"
                ]  # TODO: 选择性价比最高的（重型 -> 重型 √ | 工人 ×）
                actions[index] = attack(
                    tuple(unit["location"]), attack_loc, action_masks[index]
                )
                continue
        if task in TASK_SPACE:
            params = TASK_MAPPING[task][1](unit, path_planer, action_masks[index])
            actions[index] = TASK_MAPPING[task][0](
                unit,
                **params,
                path_planer=path_planer,
                action_mask=action_masks[index],
            )


def test_task():
    from PLAR.utils.utils import load_args, CHOOSEN_MAPS
    from PLAR.obs2text import obs_2_text
    from PLAR.text2coa import text_2_coa, subtask_assignment

    # import env and AI bots
    from gym_microrts import microrts_ai
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from stable_baselines3.common.vec_env import VecVideoRecorder

    args = load_args()

    map_name = CHOOSEN_MAPS["3"]
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.randomAI for _ in range(1)],
        map_paths=[map_name],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=False,
    )
    env.metadata["video.frames_per_second"] = args.video_fps

    name_prefix = map_name.split("maps/")[-1].split(".xml")[0].replace("/", "-")
    env = VecVideoRecorder(
        env,
        "videos",
        record_video_trigger=lambda x: True,
        video_length=args.video_length,
        name_prefix=name_prefix,
    )

    obs = env.reset()

    for i in range(600):
        print(f"{'-'*20} step-{i} {'-'*20}")
        obs_text, obs_json = obs_2_text(obs)
        # print_ai_info(i, obs, obs_json)
        actions = np.zeros((64, 7), dtype=int)
        action_mask = env.get_action_mask()
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])
        height = obs_json["env"]["height"]
        width = obs_json["env"]["width"]
        obs = obs.reshape((height, width, -1))
        valid_map = np.zeros(shape=(height, width))
        valid_map[np.where(obs[:, :, 13] == 1)] = 1  # UNIT_NONE_INDEX
        valid_map = build_place_invalid(obs, valid_map)

        path_planner = path_planning(valid_map)

        tasks = [
            "[harvest_mineral]",
            "[produce_worker]",
            "[build_barrack]",
            "[harvest_mineral]",
            "[produce_ranged]",
        ]

        tasks_params = [
            [(0, 0), (1, 0), (1, 2), (1, 1)],  # harvest_mineral
            [(2, 2)],  # produce_worker
            [(3, 2), (2, 2)],  # build_barrack
            [(0, 0), (0, 1), (1, 2), (0, 2)],  # harvest_mineral
            [(3, 4)],  # produce_ranged
        ]

        if i == 250:
            qwsw = 1
        if i == 330:
            qwsw = 1

        for task, params in zip(tasks, tasks_params):
            if task == "[harvest_mineral]":
                for i in range(len(obs_json[FIGHT_FOR]["worker"])):
                    if obs_json[FIGHT_FOR]["worker"][i]["task"] == "noop":
                        obs_json[FIGHT_FOR]["worker"][i]["task"] = task
                        index = (
                            obs_json[FIGHT_FOR]["worker"][i]["location"][0] * width
                            + obs_json[FIGHT_FOR]["worker"][i]["location"][1]
                        )
                        actions[index] = task_harvest_mineral(
                            obs_json[FIGHT_FOR]["worker"][i],
                            *params,
                            path_planner=path_planner,
                            action_mask=action_mask[index],
                        )
                        break
            elif task == "[produce_worker]":
                for i in range(len(obs_json[FIGHT_FOR]["base"])):
                    if obs_json[FIGHT_FOR]["base"][i]["task"] == "noop":
                        obs_json[FIGHT_FOR]["base"][i]["task"] = task
                        index = (
                            obs_json[FIGHT_FOR]["base"][i]["location"][0] * width
                            + obs_json[FIGHT_FOR]["base"][i]["location"][1]
                        )
                        actions[index] = task_produce_worker(
                            obs_json[FIGHT_FOR]["base"][i],
                            *params,
                            action_mask=action_mask[index],
                        )
                        break
            elif task == "[build_barrack]":
                for i in range(len(obs_json[FIGHT_FOR]["worker"])):
                    if obs_json[FIGHT_FOR]["worker"][i]["task"] == "noop":
                        obs_json[FIGHT_FOR]["worker"][i]["task"] = task
                        index = (
                            obs_json[FIGHT_FOR]["worker"][i]["location"][0] * width
                            + obs_json[FIGHT_FOR]["worker"][i]["location"][1]
                        )
                        actions[index] = task_build_barrack(
                            obs_json[FIGHT_FOR]["worker"][i],
                            *params,
                            path_planner=path_planner,
                            action_mask=action_mask[index],
                        )
                        break
            elif task == "[produce_ranged]":
                for i in range(len(obs_json[FIGHT_FOR]["barrack"])):
                    if obs_json[FIGHT_FOR]["barrack"][i]["task"] == "noop":
                        obs_json[FIGHT_FOR]["barrack"][i]["task"] = task
                        index = (
                            obs_json[FIGHT_FOR]["barrack"][i]["location"][0] * width
                            + obs_json[FIGHT_FOR]["barrack"][i]["location"][1]
                        )
                        actions[index] = task_produce_ranged(
                            obs_json[FIGHT_FOR]["barrack"][i],
                            *params,
                            action_mask=action_mask[index],
                        )
                        break

        obs, reward, done, info = env.step(np.array(actions))

        if done:
            obs = env.reset()
    env.close()


def main():
    from PLAR.utils.utils import load_args, CHOOSEN_MAPS
    from PLAR.obs2text import obs_2_text
    from PLAR.text2coa import text_2_coa, subtask_assignment

    # import env and AI bots
    from gym_microrts import microrts_ai
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from stable_baselines3.common.vec_env import VecVideoRecorder

    args = load_args()

    map_name = CHOOSEN_MAPS["3"]
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.randomAI for _ in range(1)],
        map_paths=[map_name],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=False,
    )
    env.metadata["video.frames_per_second"] = args.video_fps

    name_prefix = map_name.split("maps/")[-1].split(".xml")[0].replace("/", "-")
    env = VecVideoRecorder(
        env,
        "videos",
        record_video_trigger=lambda x: True,
        video_length=args.video_length,
        name_prefix=name_prefix,
    )

    obs = env.reset()

    for i in range(600):
        print(f"{'#'*20}step-{i}{'#'*20}")
        obs_text, obs_json = obs_2_text(obs)
        # print_ai_info(i, obs, obs_json)

        if i % 100 == 0:
            response = """
START of COA
1. [Harvest Mineral][((0, 0), (1, 0), (1, 2), (1, 1))]
2. [Build Base
3. [Build Barrack][((3, 2), (2, 2))]
4. [Produce Worker][((2, 2))]
5. [Produce Light Soldier
6. [Produce Heavy Soldier
7. [Produce Ranged Soldier][((3, 4))]
8. [Attack Enemy Worker
9. [Attack Enemy Buildings
10. [Attack Enemy Soldiers
END of COA
"""
        coa = text_2_coa(obs_json=obs_json, llm_response=response)

        for task in coa:
            obs_json = subtask_assignment(obs_json, task)
        print(f"Assigned Task: {obs_json['blue']}")

        action = script_mapping(env, obs=obs, obs_json=obs_json)

        obs, reward, done, info = env.step(np.array(action))

        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    # main()
    test_task()
