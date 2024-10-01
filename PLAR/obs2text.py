import numpy as np
import json


def get_json(obs: np.ndarray, resources: np.ndarray) -> dict:

    UNIT_ID_INDEX = 0
    HP_INDEX = 1
    RESOURCE_INDEX = 2
    OWNER_INDEX = 3
    UNIT_TYPE_INDEX = 4
    CUR_ACTION_INDEX = 5

    ACTION_MAP = {
        '0': 'noop',
        '1': 'move',
        '2': 'harvest',
        '3': 'return',
        '4': 'produce',
        '5': 'attack'
    }

    UNIT_TYPE_MAP = {
        "resource": 1,
        "base": 2,
        "barrack": 3,
        "worker": 4,
        "light": 5,
        "heavy": 6,
        "ranged": 7,
    }

    OWNER_MAP = {"blue": 1, "red": 2}

    data_json = {}

    # 环境
    obs = np.squeeze(obs)
    resources = np.squeeze(resources)
    data_json['env'] = {}
    data_json['env']['height'] = obs.shape[1]
    data_json['env']['width'] = obs.shape[2]
    data_json['units'] = {}
    for i in range(data_json['env']['height']):
        for j in range(data_json['env']['width']):
            data_json['units'][(i,j)] = {}

    locations = np.where(obs[UNIT_TYPE_INDEX] == UNIT_TYPE_MAP['resource'])
    locations = np.array(locations).T
    data_json['env']['resource'] = []
    for location in locations:
        location = tuple(location.tolist())
        d = {
            "owner": "env",
            "type": "resource",
            "location": location,
            "resource_num": int(obs[RESOURCE_INDEX][location]),
            "id": int(obs[UNIT_ID_INDEX][location])
        }
        data_json['env']['resource'].append(d)
        data_json['units'][location] = d

    def get_unit_json(obs, obs_json, unit_type, owner):
        locations = np.where(
            (
                (obs[UNIT_TYPE_INDEX] == UNIT_TYPE_MAP[unit_type]) 
                & (obs[OWNER_INDEX] == OWNER_MAP[owner])
            )
        )
        locations = np.array(locations).T
        obs_json[owner][unit_type] = []
        for location in locations:
            location = tuple(location.tolist())
            d = {
                "owner": owner,
                "type": unit_type,
                "location": location,
                "hp": int(obs[HP_INDEX][location]),
                "action": ACTION_MAP[str(obs[CUR_ACTION_INDEX][location])],
                "id": int(obs[UNIT_ID_INDEX][location]),
                "resource_num": int(obs[RESOURCE_INDEX][location]),
                "task": "[noop]",
                "task params": (),
            }
            obs_json[owner][unit_type].append(d)
            obs_json["units"][location] = d
        return obs_json

    for owner in OWNER_MAP.keys():
        data_json[owner] = {}
        data_json["blue"]["resource"] = int(resources[OWNER_MAP[owner] - 1])
        for unit_type in ["base", "barrack", "worker", "light", "heavy", "ranged"]:
            data_json = get_unit_json(obs, data_json, unit_type, owner)

    return data_json


CONST_BLUE = 'blue team'
CONST_RED = 'red team'

CONST_MINERAL = 'mineral field'
CONST_BASE = 'base'
CONST_BARRACK = 'barrack'
CONST_WORKER = 'worker'
CONST_LIGHT = 'light soldier'
CONST_HEAVY = 'heavy soldier'
CONST_RANGED = 'ranged soldier'

def get_env_text(env_data: dict) -> str:
    '''
    input: data['env']
    {
        resource: [
            resource1{location:xx, resource_num:xx}, 
            resource2{}, 
            ...
        ]
    }
    output: text description of env
    '''
    text = ''
    resources = env_data['resource']
    if len(resources) == 0:
        text = f"There are no {CONST_MINERAL} available on this map. "
    elif len(resources) == 1:
        text = f"There is one {CONST_MINERAL} located in {resources[0]['location']} with {resources[0]['resource_num']} available resources. "
    else:
        text = f"There are {len(resources)} {CONST_MINERAL}s in this map. "
        for i in range(len(resources)):
            # if resources[i]['resource_num'] == 4:
            #     text += f"The {CONST_MINERAL} located in {resources[i]['location']} has at least 4 resources. "
            # else:
            text += f"The {CONST_MINERAL} located in {resources[i]['location']} has {resources[i]['resource_num']} available resources. "

    return  text


def get_blue_text(blue_data: dict) -> str:
    '''
    input: data['blue']
    {
        'base':[{location:xx, hp:xx, resource_num:xx, action:xx}],
        'barrack':[{location:xx, hp:xx, action:xx}],
        'worker':[{location:xx, hp:xx, resource_num:xx, action:xx}],
        'light':[{location:xx, hp:xx, action:xx}],
        'heavy':[{location:xx, hp:xx, action:xx}],
        'ranged':[{location:xx, hp:xx, action:xx}]
    }
    output: text description of env
    '''

    base = blue_data['base']
    text_base = ''
    if len(base) == 0:
        text_base = f"The {CONST_BLUE} has no {CONST_BASE}. "
    elif len(base) == 1:
        text_base = f"The {CONST_BLUE} has one {CONST_BASE} located in {base[0]['location']} with {base[0]['resource_num']} remaining resource, {base[0]['hp']} remaining HP, and the current action of it is {base[0]['action']}. "
    else:
        text_base = f"The {CONST_BLUE} has a total of {len(base)} {CONST_BASE}s. "
        for i in range(len(base)):
            text_base += f"The {CONST_BASE} located in {base[i]['location']} has {base[i]['resource_num']} remaining resource, {base[i]['hp']} remaining HP, and the current action of it is {base[i]['action']}. "

    worker = blue_data['worker']
    text_worker = ''
    if len(worker) == 0:
        text_worker = f"The {CONST_BLUE} has no {CONST_WORKER}. "
    elif len(worker) == 1:
        text_worker = f"The {CONST_BLUE} has one {CONST_WORKER} located in {worker[0]['location']}, which carries {worker[0]['resource_num']} resource and the current action is {worker[0]['action']}. "
    else:
        text_worker = f"The {CONST_BLUE} has a total of {len(worker)} {CONST_WORKER}s. "
        for i in range(len(worker)):
            text_worker += f"The {CONST_WORKER} located in {worker[i]['location']} carries {worker[i]['resource_num']} resource and the current action is {worker[i]['action']}. "

    barrack = blue_data['barrack']
    text_barrack = ''
    if len(barrack) == 0:
        text_barrack = f"The {CONST_BLUE} has no {CONST_BARRACK}. "
    elif len(barrack) == 1:
        text_barrack = f"The {CONST_BLUE} has one {CONST_BARRACK} located in {barrack[0]['location']}, which has {barrack[0]['hp']} HP and the current action is {barrack[0]['action']}. "
    else:
        text_barrack = f"The {CONST_BLUE} has a total of {len(barrack)} {CONST_BARRACK}s. "
        for i in range(len(barrack)):
            text_barrack += f"The {CONST_BARRACK} located in {barrack[i]['location']} has {barrack[i]['hp']} HP and the current action is {barrack[i]['action']}. "

    light = blue_data['light']
    text_light = ''
    if len(light) == 0:
        text_light = f"The {CONST_BLUE} has no {CONST_LIGHT}. "
    elif len(light) == 1:
        text_light = f"The {CONST_BLUE} has one {CONST_LIGHT} located in {light[0]['location']}, which has {light[0]['hp']} HP and the current action is {light[0]['action']}. "
    else:
        text_light = f"The {CONST_BLUE} has a total of {len(light)} {CONST_LIGHT}s. "
        for i in range(len(light)):
            text_light += f"The {CONST_LIGHT} located in {light[i]['location']} has {light[i]['hp']} HP and the current action is {light[i]['action']}. "

    heavy = blue_data['heavy']
    text_heavy = ''
    if len(heavy) == 0:
        text_heavy = f"The {CONST_BLUE} has no {CONST_HEAVY}. "
    elif len(heavy) == 1:
        text_heavy = f"The {CONST_BLUE} has one {CONST_HEAVY} located in {heavy[0]['location']}, which has {heavy[0]['hp']} HP and the current action is {heavy[0]['action']}. "
    else:
        text_heavy = f"The {CONST_BLUE} has a total of {len(heavy)} {CONST_HEAVY}s. "
        for i in range(len(heavy)):
            text_heavy += f"The {CONST_HEAVY} located in {heavy[i]['location']} has {heavy[i]['hp']} HP and the current action is {heavy[i]['action']}. "

    ranged = blue_data['ranged']
    text_ranged = ''
    if len(ranged) == 0:
        text_ranged = f"The {CONST_BLUE} has no {CONST_RANGED}. "
    elif len(ranged) == 1:
        text_ranged = f"The {CONST_BLUE} has one {CONST_RANGED} located in {ranged[0]['location']}, which has {ranged[0]['hp']} HP and the current action is {ranged[0]['action']}. "
    else:
        text_ranged = f"The {CONST_BLUE} has a total of {len(ranged)} {CONST_RANGED}s. "
        for i in range(len(ranged)):
            text_ranged += f"The {CONST_RANGED} located in {ranged[i]['location']} has {ranged[i]['hp']} HP and the current action is {ranged[i]['action']}. "

    text = text_base + text_barrack + text_worker + \
        text_light + text_heavy + text_ranged
    return text


def get_red_text(red_data: dict) -> str:
    '''
    input: data['red']
    output: text description of env
    '''
    return get_blue_text(red_data).replace(CONST_BLUE, CONST_RED)


from typing import Tuple


def obs_2_text(obs: Tuple[np.ndarray, np.ndarray], zh=False) -> Tuple[str, dict]:
    '''
    Input: 
        np.ndarray: observation

    Output: 
        str: text description of observation
        dict: json type observation
    '''
    # print(f"{'*'*10}Obs2Text: running{'*'*10}", flush=True)
    resources, obs = obs
    map_height = obs.shape[1]
    map_width = obs.shape[2]
    text = ''
    obs = obs.reshape((map_height, map_width, -1))
    resources = resources.reshape(-1)

    data = get_json(obs)
    # print(data)

    with open('data.json', 'w') as f:
        import copy
        writable_data = copy.deepcopy(data)
        # key with tuple type cant be dumped
        writable_data['units'] = dict((str(k), v) for k,v in data['units'].items())
        f.write(json.dumps(writable_data))

    # env
    text_env = get_env_text(data['env'])
    text_red = get_red_text(data['red'])
    text_blue = get_blue_text(data['blue'])

    text = text_env + text_red + text_blue
    # print(f"Observation Text: \n{text}")

    if zh:
        from PLAR.utils.utils import en2zh
        text_ZH = en2zh(text)
        # print(f"Observation Text_ZH: \n{text_ZH}")

    # print(f"{'*'*10}Obs2Text: done{'*'*10}", flush=True)
    return text, data


def show_all_maps_figure():
    '''
    visualize all maps, located in `show_maps` directory
    '''

    import os
    # import env and AI bots
    from gym_microrts import microrts_ai
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from stable_baselines3.common.vec_env import VecVideoRecorder

    map_dir = "/root/desc/play-urts/gym_microrts/microrts/maps"
    
    all_map_list = []
    for path, __, files in os.walk(map_dir):
        for file_name in files:
            all_map_list.append(os.path.join(path, file_name))
    all_map_list = sorted(all_map_list)

    all_map_list = all_map_list[1:] # remove /maps/.DS_Store

    print(",\n".join(all_map_list))
    
    for tt in all_map_list:
        map_name = tt[tt.find('maps/'):]
        envs = MicroRTSGridModeVecEnv(
            num_selfplay_envs=0,
            num_bot_envs=1,
            max_steps=2000,
            render_theme=2,
            ai2s=[microrts_ai.coacAI for _ in range(1)],
            map_paths=[map_name],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            # autobuild=False
        )
        name_prefix = map_name.split('maps/')[-1].split('.xml')[0].replace('/', '-')

        envs = VecVideoRecorder(envs, "show_maps", record_video_trigger=lambda x: x == 0, video_length=1, name_prefix=name_prefix)
        obs = envs.reset()
        envs.close()

    os.system("rm /root/desc/play-urts/PLAR/show_maps/*.json")


def test_obs_2_text():
    # import env and AI bots
    from gym_microrts import microrts_ai
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from stable_baselines3.common.vec_env import VecVideoRecorder

    from PLAR.utils.utils import CHOSEN_MAPS
    print(len(CHOSEN_MAPS), CHOSEN_MAPS)

    for map_name in CHOSEN_MAPS.values():
        envs = MicroRTSGridModeVecEnv(
            num_selfplay_envs=0,
            num_bot_envs=1,
            max_steps=2000,
            render_theme=2,
            ai2s=[microrts_ai.coacAI for _ in range(1)],
            map_paths=[map_name],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            autobuild=False
        )
        name_prefix = map_name.split('maps/')[-1].split('.xml')[0].replace('/', '-')
        # envs = VecVideoRecorder(envs, "videos", record_video_trigger=lambda x: x == 0, video_length=1, name_prefix=name_prefix)
        
        obs = envs.reset()
        # print(obs.shape)
        # (1, width, height, 27)

        obs_text, obs_json = obs_2_text(obs)
        with open('./texts/' + name_prefix, 'w') as f:
            f.write(json.dumps(obs_json) + '\n')
            f.write(obs_text + '\n')

        envs.close()


if __name__ == "__main__":
    # show_all_maps_figure()
    test_obs_2_text()
    exit()
