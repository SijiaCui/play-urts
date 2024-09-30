import copy
import numpy as np

from PLAR.script_mapping import script_mapping
from PLAR.utils.utils import CHOSEN_MAPS, FIGHT_FOR, ENEMY, parse_task, load_args
from PLAR.obs2text import get_json, obs_2_text
from PLAR.utils.llm_agent import coa_agent

# import env and AI bots
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder


def main():
    args = load_args()
    agent = coa_agent(args)

    map_name = CHOSEN_MAPS["3"]
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(1)],
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

    situation = {
        "blue": {
            "worker": 1,
            "base": 1,
            "barrack": 0,
            "light": 0,
            "heavy": 0,
            "ranged": 0,
        },
        "red": {
            "worker": 0,
            "base": 0,
            "barrack": 0,
            "light": 0,
            "heavy": 0,
            "ranged": 0,
        },
    }
    task_list, task_params = [], []
    for i in range(int(1e9)):
        print(f"{'-'*20} step-{i} {'-'*20}")

        obs_text, obs_dict = obs_2_text(obs)

        if i % 1000 == 0:
            # response = agent.run(obs_text)
            response = """
START of TASK
[Harvest Mineral]((0, 0), (1, 0), (1, 2), (1, 1)),
[Produce Unit]('worker', 'south'),
[Build Building]('barrack', (3, 2), (2, 2)),
[Harvest Mineral]((0, 0), (0, 1), (1, 2), (0, 2))],
[Produce Unit]('ranged', 'east'),
[Produce Unit]('light', 'south'),
[Produce Unit]('ranged', 'south'),
[Produce Unit]('heavy', 'east'),
[Attack Enemy]('light', 'base', (6, 5), (5, 5)),
[Attack Enemy]('ranged', 'worker', (6, 6), (4, 5)),
[Attack Enemy]('heavy', 'base', (6, 7), (5, 7)),
[Attack Enemy]('heavy', 'barrack', (6, 7), (5, 7)),
END of TASK
"""
            task_list, task_params = parse_task(response)

        old_situation = copy.deepcopy(situation)
        situation = update_situation(situation, obs_dict)
        task_list, task_params = update_task_list(task_list, task_params, situation, old_situation)

        action_vectors = script_mapping(env, task_list, task_params, obs_dict)

        obs, reward, done, info = env.step(np.array(action_vectors))

        if done:
            env.close()
            print(f"Game over, reward: {reward}")
            break


def update_situation(situation, obs_dict):
    for keys in situation[FIGHT_FOR].keys():
        situation[FIGHT_FOR][keys] = len(obs_dict[FIGHT_FOR][keys])
    for keys in situation[ENEMY].keys():
        situation[ENEMY][keys] = len(obs_dict[ENEMY][keys])
    return situation


def update_task_list(task_list, task_params, new_situation, old_situation):
    for unit_type in new_situation[FIGHT_FOR].keys():
        changes = max(new_situation[FIGHT_FOR][unit_type] - old_situation[FIGHT_FOR][unit_type], 0)
        for j, (task, params) in enumerate(zip(task_list, task_params)):
            if (
                params[0] == unit_type
                and task
                in [
                    "[Produce Unit]",
                    "[Build Building]",
                ]
                and changes > 0
            ):
                task_list.pop(j)
                task_params.pop(j)
                changes -= 1

    for unit_type in new_situation[ENEMY].keys():
        changes = max(old_situation[ENEMY][unit_type] - new_situation[ENEMY][unit_type], 0)
        for j, (task, params) in enumerate(zip(task_list, task_params)):
            if params[1] == unit_type and task in ["[Attack Enemy]"] and changes > 0:
                task_list.pop(j)
                task_params.pop(j)
                changes -= 1

    return task_list, task_params


if __name__ == "__main__":
    main()
