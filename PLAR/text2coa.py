import numpy as np

def parse_coa(text: str) -> list:
    import re
    pattern = r'^(.*)\[(.+)\](.*)$'
    
    from PLAR.utils.utils import COA_ACTION_SPACE
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
    except Exception as e:
        print(f"Response Processing Error: {e}")
    return coa_list


def assign_task(obs_json: dict, task: str) -> dict:
    from PLAR.utils.utils import COA_ACTION_SPACE
    assert task in COA_ACTION_SPACE

    FIGHT_FOR = 'blue'

    from PLAR.utils.utils import COA_H_Mineral, COA_B_Base, COA_B_Barrack, COA_P_Worker, COA_P_Light, COA_P_Heavy, COA_P_Ranged, COA_A_Worker, COA_A_Buildings, COA_A_Soldiers
    
    def h_mineral(obs, task) -> dict:
        # 考虑的因素：哪个worker离mineral的距离近
        is_assigned = -1
        for i in range(len(obs[FIGHT_FOR]['worker'])):
            if obs[FIGHT_FOR]['worker'][i]['action'] == 'noop' and \
                obs[FIGHT_FOR]['worker'][i]['task'] == 'noop':
                obs[FIGHT_FOR]['worker'][i]['task'] = task
                is_assigned = i
                break

        return obs, is_assigned
    
    def b_base(obs, task) -> dict:
        is_assigned = -1
        for i in range(len(obs[FIGHT_FOR]['worker'])):
            if obs[FIGHT_FOR]['worker'][i]['action'] == 'noop' and \
                obs[FIGHT_FOR]['worker'][i]['task'] == 'noop':
                obs[FIGHT_FOR]['worker'][i]['task'] = task
                is_assigned = i
                break

        return obs, is_assigned
    
    def b_barrack(obs, task) -> dict:
        is_assigned = -1
        for i in range(len(obs[FIGHT_FOR]['worker'])):
            if obs[FIGHT_FOR]['worker'][i]['action'] == 'noop' and \
                obs[FIGHT_FOR]['worker'][i]['task'] == 'noop':
                obs[FIGHT_FOR]['worker'][i]['task'] = task
                is_assigned = i
                break

        return obs, is_assigned
    
    def p_worker(obs, task) -> dict:
        is_assigned = -1
        for i in range(len(obs[FIGHT_FOR]['base'])):
            if obs[FIGHT_FOR]['base'][i]['action'] == 'noop' and \
                obs[FIGHT_FOR]['base'][i]['task'] == 'noop':
                obs[FIGHT_FOR]['base'][i]['task'] = task
                is_assigned = i
                break

        return obs, is_assigned

    def p_light(obs, task) -> dict:
        is_assigned = -1
        for i in range(len(obs[FIGHT_FOR]['barrack'])):
            if obs[FIGHT_FOR]['barrack'][i]['action'] == 'noop' and \
                obs[FIGHT_FOR]['barrack'][i]['task'] == 'noop':
                obs[FIGHT_FOR]['barrack'][i]['task'] = task
                is_assigned = i
                break

        return obs, is_assigned
    
    def p_heavy(obs, task) -> dict:
        return p_light(obs, task)
    
    def p_ranged(obs, task) -> dict:
        return p_light(obs, task)
    
    def a_worker(obs, task) -> dict:
        is_assigned = -1

        # num_worker = len(obs[FIGHT_FOR]['worker'])
        num_light = len(obs[FIGHT_FOR]['light'])
        num_heavy = len(obs[FIGHT_FOR]['heavy'])
        num_ranged = len(obs[FIGHT_FOR]['ranged'])
        
        for i in range(num_light):
            if obs[FIGHT_FOR]['light'][i]['action'] == 'noop' and \
                obs[FIGHT_FOR]['light'][i]['task'] == 'noop':
                obs[FIGHT_FOR]['light'][i]['task'] = task
        for i in range(num_heavy):
            if obs[FIGHT_FOR]['heavy'][i]['action'] == 'noop' and \
                obs[FIGHT_FOR]['heavy'][i]['task'] == 'noop':
                obs[FIGHT_FOR]['heavy'][i]['task'] = task
        for i in range(num_ranged):
            if obs[FIGHT_FOR]['ranged'][i]['action'] == 'noop' and \
                obs[FIGHT_FOR]['ranged'][i]['task'] == 'noop':
                obs[FIGHT_FOR]['ranged'][i]['task'] = task

        return obs, is_assigned
    
    def a_buildings(obs, task) -> dict:
        return a_worker(obs, task)
    
    def a_soldiers(obs, task) -> dict:
        return a_worker(obs, task)

    TASK_ASSIGNMENT_MAP = {
        COA_H_Mineral: h_mineral,
        COA_B_Base: b_base,
        COA_B_Barrack: b_barrack,
        COA_P_Worker: p_worker,
        COA_P_Light: p_light,
        COA_P_Heavy: p_heavy,
        COA_P_Ranged: p_ranged,
        COA_A_Worker: a_worker,
        COA_A_Buildings: a_buildings,
        COA_A_Soldiers: a_soldiers
    }
    
    obs_json: dict = TASK_ASSIGNMENT_MAP[task](obs_json, task)[0]
    return obs_json


from typing import List, Tuple 
def text_2_coa(obs_json: dict, llm_response: str) -> Tuple[List[str], dict]:
    '''
    Input: 
        dict: json type observation
        str: coa response of LLM

    Output:
        List(str): parsed coa
        dict: json type observation with assigned task
    '''
    print(f"{'*'*10}Text2COA: running{'*'*10}", flush=True)
    coa_list = parse_coa(llm_response)
    print(f"Parsed COA: {coa_list}")

    # task assignment
    for task in coa_list:
        obs_json = assign_task(obs_json, task)
    
    print(f"Assigned Task: {obs_json['blue']}")
    print(f"{'*'*10}Text2COA: done{'*'*10}", flush=True)
    return coa_list, obs_json


def test_text_2_coa():
    from PLAR.llm_agent import coa_agent
    from PLAR.utils.utils import load_args, CHOOSEN_MAPS
    from PLAR.obs2text import obs_2_text

    # import env and AI bots
    from gym_microrts import microrts_ai
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from stable_baselines3.common.vec_env import VecVideoRecorder

    args = load_args()
    ca = coa_agent(args)

    map_name = CHOOSEN_MAPS['1']
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
    obs_text, obs_json = obs_2_text(obs)
    response = ca.run(obs_text)

    print(f"Before Assignment: {obs_json}")
    coa_list, obs_json = text_2_coa(obs_json, response)
    print(f"COA: {coa_list}\nAfter Assignment: {obs_json}")

    env.close()


if __name__ == "__main__":
    test_text_2_coa()
    exit()
