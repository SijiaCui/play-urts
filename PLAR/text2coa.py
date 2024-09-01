import numpy as np
from queue import Queue

from PLAR.utils.utils import COA_ACTION_SPACE, sample_action, COA_SCRIPT_MAPPING

def parse_coa(text: str) -> list:
    import re
    pattern = r'^(.*)\[(.+)\](.*)$'
    
    coa_list = []
    try:
        text = text.split('START of COA')[1].split('END of COA')[0]
        text_list = text.split('\n')
        for coa in text_list:
            # match = re.match(pattern, coa)
            # if match and '['+match.group(2)+']' in COA_ACTION_SPACE:
            #     coa_list.append('['+match.group(2)+']')
            # else:
            #     print(f"failed: {coa}")
            sbeg = coa.find('[')
            send = coa.find(']')
            if  sbeg + 1 and send + 1 and coa[sbeg: send + 1] in COA_ACTION_SPACE:
                coa_list.append(coa[sbeg: send + 1])
            elif coa != '':
                print(f"pass: {coa}")
    except:
        print("Response Processing Error")
    print(f"Parsed COA: {coa_list}")
    return coa_list


def text_2_coa(action_queue: Queue, env, obs: np.ndarray, text: str) -> Queue:
    '''
    COA -> action queue
    '''
    coa_list = parse_coa(text)

    #TODO
    # coa_list -> action list

    # action = sample_action(env)
    for coa in coa_list:
        action_list = COA_SCRIPT_MAPPING[coa](env, obs)
        for action in action_list:
            action_queue.put(action)

    return action_queue


if __name__ == "__main__":
    text = """
    START of COA
    1. [Attack Enemy Buildings]
    2. [Produce Worker]
    3. ...
    END of COA
    """
    print(f"origin COA text: {text}")
    aq = text_2_coa(text)
    print(list(aq.queue))
    exit()

    from PLAR.llm_agent import coa_agent
    from PLAR.utils.utils import load_args, CHOOSEN_MAPS
    from PLAR.obs2text import obs_2_text

    # import env and AI bots
    from gym_microrts import microrts_ai
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from stable_baselines3.common.vec_env import VecVideoRecorder

    args = load_args()
    ca = coa_agent(args)

    action_queue = Queue(100)

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
                action_queue = Queue(100)
