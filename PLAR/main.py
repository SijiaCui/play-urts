import numpy as np

from PLAR.utils.llm_agent import coa_agent
from PLAR.utils.utils import load_args, CHOSEN_MAPS, parse_task, path_planning

from PLAR.obs2text import obs_2_text
# from PLAR.text2coa import text_2_coa, subtask_assignment

# import env and AI bots
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder


def main():
    args = load_args()
    ca = coa_agent(args)

    map_name = CHOSEN_MAPS['3']
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.randomAI for _ in range(1)],
        map_paths=[map_name],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=False
    )
    env.metadata['video.frames_per_second'] = args.video_fps

    name_prefix = map_name.split('maps/')[-1].split('.xml')[0].replace('/', '-')
    env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: True, video_length=args.video_length, name_prefix=name_prefix)

    obs = env.reset()

    for i in range(600):
        print(f"{'#'*20}step-{i}{'#'*20}")
        obs_text, obs_json = obs_2_text(obs)

        if i % 100 == 0:
            # response = ca.run(obs_text)
            response = """
# START of COA
# 1. [Harvest Mineral]
# 2. [Build Base
# 3. [Build Barrack
# 4. [Produce Worker]
# 5. [Produce Light Soldier
# 6. [Produce Heavy Soldier
# 7. [Produce Ranged Soldier
# 8. [Attack Enemy Worker]
# 9. [Attack Enemy Buildings]
# 10. [Attack Enemy Soldiers
# END of COA
# """
        height = obs_json["env"]["height"]
        width = obs_json["env"]["width"]
        action_mask = env.get_action_mask()
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])

        # generate a valid map that indicates that grid is valid to be moved on
        obs = obs.reshape((height, width, -1))
        valid_map = np.zeros(shape=(height, width))
        valid_map[np.where(obs[:, :, 13] == 1)] = 1  # UNIT_NONE_INDEX

        path_planer = path_planning(valid_map)
        print(path_planer.get_locs_with_dist_from_tgt((1, 1), 3))

    # for task in coa:
    #     obs_json = subtask_assignment(obs_json, task)
    # print(f"Assigned Task: {obs_json['blue']}")

    # action = script_mapping(env, obs=obs, obs_json=obs_json)

    # obs, reward, done, info = env.step(np.array(action))

    # if done:
    #     obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()
