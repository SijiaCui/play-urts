import numpy as np
from typing import Tuple, List

from PLAR.script_mapping import script_mapping
from PLAR.utils.utils import CHOSEN_MAPS, parse_task, load_args
from PLAR.obs2text import obs_2_text
from PLAR.utils.llm_agent import LLMAgent
from PLAR.task2actions import (
    TASK_HARVEST_MINERAL,
    TASK_BUILD_BUILDING,
    TASK_PRODUCE_UNIT,
    TASK_ATTACK_ENEMY,
)
import PLAR.utils.utils as utils
from PLAR.utils.map_info import MAP_INFO

# import env
from gym_microrts.envs.plar_vec_env import MicroRTSGridModePLARVecEnv
from gym_microrts.envs.plar_vec_video_recorder import PLARVecVideoRecorder


def main():
    # ====================
    #        Init
    # ====================
    args = load_args()
    llm_agents = [LLMAgent(args), LLMAgent(args)]

    map_name = CHOSEN_MAPS[str(args.map_index)]
    situation = MAP_INFO[args.map_index]
    env = MicroRTSGridModePLARVecEnv(
        num_selfplay_envs=2,
        num_bot_envs=0,
        max_steps=args.max_steps,
        map_paths=[map_name],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=False,
    )
    env.metadata["video.frames_per_second"] = args.video_fps
    name_prefix = map_name.split("maps/")[-1].split(".xml")[0].replace("/", "-")
    env = PLARVecVideoRecorder(env, "videos", lambda x: True, args.video_length, name_prefix)
    obs = env.reset()
    blue_tasks, red_tasks = [], []

    # ====================
    #        Playing
    # ====================
    for i in range(args.max_steps):
        print(f"{'-'*20} step-{i} {'-'*20}")
        blue_action_vectors, blue_tasks = get_action_vectors(i, env, obs, situation, args.tasks_update_interval, blue_tasks, llm_agents[0], "blue")
        red_action_vectors, red_tasks = get_action_vectors(i, env, obs, situation, args.tasks_update_interval, red_tasks, llm_agents[1], "red")
        obs, reward, done, info = env.step(np.array([blue_action_vectors, red_action_vectors]))

        if done[0]:
            env.close()
            print(f"Game over, reward: {reward}")
            break


def update_situation(situation, obs_dict):
    def update_unit_count(situation_section, obs_section):
        for unit_type in situation_section.keys():
            for unit in obs_section.values():
                if isinstance(unit, dict) and unit["type"] == unit_type:
                    situation_section[unit_type] += 1

    new_situation = {}
    new_situation[utils.FIGHT_FOR] = {
        unit_type: 0 for unit_type in situation[utils.FIGHT_FOR].keys()
    }
    new_situation[utils.ENEMY] = {
        unit_type: 0 for unit_type in situation[utils.ENEMY].keys()
    }

    update_unit_count(new_situation[utils.FIGHT_FOR], obs_dict[utils.FIGHT_FOR])
    update_unit_count(new_situation[utils.ENEMY], obs_dict[utils.ENEMY])

    return new_situation, situation


def update_tasks_by_situation(tasks: List[Tuple], situation, obs_dict):
    new_situation, old_situation = update_situation(situation, obs_dict)

    def process_tasks(tasks, situation_key, unit_index, task_types, changes_condition):
        for unit_type in new_situation[situation_key].keys():
            changes = max(changes_condition(unit_type), 0)
            for task in tasks:
                if (
                    task[0] in task_types
                    and task[1][unit_index] == unit_type
                    and changes > 0
                ):
                    print(f"Completed task: {task[0]}{task[1]}")
                    tasks.remove(task)
                    changes -= 1
                    if changes == 0:
                        break

    process_tasks(
        tasks,
        utils.FIGHT_FOR,
        0,
        [TASK_PRODUCE_UNIT, TASK_BUILD_BUILDING],
        lambda unit_type: new_situation[utils.FIGHT_FOR][unit_type]
        - old_situation[utils.FIGHT_FOR][unit_type],
    )

    process_tasks(
        tasks,
        utils.ENEMY,
        1,
        [TASK_ATTACK_ENEMY],
        lambda unit_type: old_situation[utils.ENEMY][unit_type]
        - new_situation[utils.ENEMY][unit_type],
    )

    for task in tasks:
        if task[0] == TASK_HARVEST_MINERAL:
            is_unit_empty = not obs_dict["units"][task[1]]
            is_unit_not_resource = obs_dict["units"][task[1]].get("type") != "resource"
            is_base_zero = new_situation[utils.FIGHT_FOR]["base"] == 0

            if is_unit_empty or is_unit_not_resource or is_base_zero:
                tasks.remove(task)

    return tasks, new_situation


def get_action_vectors(step, env, obs, situation, interval, tasks, llm_agent, fight_for):
    obs_text, obs_dict = obs_2_text(obs[0])
    if fight_for == "blue":
        utils.FIGHT_FOR = "blue"
        utils.ENEMY = "red"
    else:
        utils.FIGHT_FOR = "red"
        utils.ENEMY = "blue"

    if step % interval == 0:
        response = llm_agent.run(obs_text)
        tasks = parse_task(response)
    tasks, situation = update_tasks_by_situation(tasks, situation, obs_dict)
    action_vectors = script_mapping(env, tasks, obs_dict, True)
    return action_vectors, tasks


if __name__ == "__main__":
    main()
