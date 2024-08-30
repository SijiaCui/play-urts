import numpy as np

from PLAR.llm_agent import coa_agent
from PLAR.utils.utils import load_args, CHOOSEN_MAPS

from PLAR.obs2text import obs_2_text
from PLAR.text2coa import text_2_coa

# import env and AI bots
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder

import os


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def sample(logits):
    # https://stackoverflow.com/a/40475357/6611317
    p = softmax(logits, axis=1)
    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    return choices.reshape(-1, 1)

def main():
    args = load_args()

    ca = coa_agent(args)

    map_name = CHOOSEN_MAPS['2']
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
    env.metadata['video.frames_per_second'] = 10
    args.video_length = 200

    name_prefix = map_name.split('maps/')[-1].split('.xml')[0].replace('/', '-')
    env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: True, video_length=args.video_length, name_prefix=name_prefix)
    xx = 0

    env.reset()
    for i in range(1000):
        env.render()
        action_mask = env.get_action_mask()
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])
        action_mask[action_mask == 0] = -9e8
        # sample valid actions
        action = np.concatenate(
            (
                sample(action_mask[:, 0:6]),  # action type
                sample(action_mask[:, 6:10]),  # move parameter
                sample(action_mask[:, 10:14]),  # harvest parameter
                sample(action_mask[:, 14:18]),  # return parameter
                sample(action_mask[:, 18:22]),  # produce_direction parameter
                sample(action_mask[:, 22:29]),  # produce_unit_type parameter
                # attack_target parameter
                sample(action_mask[:, 29 : sum(env.action_space.nvec[1:])]),
            ),
            axis=1,
        )
        # doing the following could result in invalid actions
        # action = np.array([envs.action_space.sample()])
        next_obs, reward, done, info = env.step(action)

        if done:
            break
            env.reset()

    env.close()


if __name__ == "__main__":
    main()