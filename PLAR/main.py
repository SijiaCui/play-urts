import numpy as np
from queue import Queue

from PLAR.llm_agent import coa_agent
from PLAR.utils.utils import load_args, CHOOSEN_MAPS, sample_action

from PLAR.obs2text import obs_2_text
from PLAR.text2coa import text_2_coa

# import env and AI bots
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder

def main():
    args = load_args()
    ca = coa_agent(args)

    action_queue = Queue(args.action_queue_size)

    map_name = CHOOSEN_MAPS[args.map_index]
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(1)],
        map_paths=[map_name],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=False
    )
    env.metadata['video.frames_per_second'] = args.video_fps

    name_prefix = map_name.split('maps/')[-1].split('.xml')[0].replace('/', '-')
    env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: True, video_length=args.video_length, name_prefix=name_prefix)
    obs = env.reset()

    for i in range(10000):
        obs_text = obs_2_text(obs)
        response = ca.run(obs_text)
        
        action_queue = text_2_coa(action_queue, env, obs, response)
        
        while not action_queue.empty():

            action = action_queue.get()
            obs, reward, done, info = env.step(action)

            if done:
                obs = env.reset()
                action_queue = Queue(args.action_queue_size)
    env.close()


if __name__ == "__main__":
    main()