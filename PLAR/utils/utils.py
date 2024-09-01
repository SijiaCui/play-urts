from PLAR.utils.fewshots import *
from PLAR.utils.prompts import *
from PLAR.utils.scripts import *

import argparse
import json
import numpy as np

# {[Harvest Mineral], [Build Base], [Build Barrack], [Produce Worker], [Produce Light Soldier], [Produce Heavy Soldier], [Produce Ranged Soldier], [Attack Enemy Worker], [Attack Enemy Buildings], [Attack Enemy Soldiers]}
COA_H_Mineral = "[Harvest Mineral]"
COA_B_Base = "[Build Base]"
COA_B_Barrack = "[Build Barrack]"

COA_P_Worker = "[Produce Worker]"
COA_P_Light = "[Produce Light Soldier]"
COA_P_Heavy = "[Produce Heavy Soldier]"
COA_P_Ranged = "[Produce Ranged Soldier]"

COA_A_Worker = "[Attack Enemy Worker]"
COA_A_Buildings = "[Attack Enemy Buildings]"
COA_A_Soldiers = "[Attack Enemy Soldiers]"

COA_ACTION_SPACE = [COA_H_Mineral, COA_B_Base, COA_B_Barrack, COA_P_Worker, COA_P_Light, COA_P_Heavy, COA_P_Ranged, COA_A_Worker, COA_A_Buildings, COA_A_Soldiers]
COA_ACTION_SPACE_STR = f"{{{', '.join(COA_ACTION_SPACE)}}}"

COA_SCRIPT_MAPPING = {
    COA_H_Mineral: harvest_mineral,

    COA_B_Base: build_base,
    COA_B_Barrack: build_barrack,

    COA_P_Worker: produce_worker,
    COA_P_Light: produce_light_soldier,
    COA_P_Heavy: produce_heavy_soldier,
    COA_P_Ranged: produce_ranged_soldier,

    COA_A_Worker: attack_enemy_worker,
    COA_A_Buildings: attack_enemy_buildings,
    COA_A_Soldiers: attack_enemy_soldiers
}


def load_args():
    parser = argparse.ArgumentParser()

    # load config file to add default arguments
    with open('configs.json', 'r') as f:
        config = json.load(f)

    # llm parameters
    parser.add_argument('--engine', type=str, default=config['llm_engine'])
    parser.add_argument('--temperature', type=float, default=float(config['llm_engine_temperature']))
    parser.add_argument('--max_tokens', type=int, default=int(config['llm_engine_max_tokens']))
    
    # video recorder parameters
    parser.add_argument('--video_fps', type=int, default=int(config['video_fps']))
    parser.add_argument('--video_length', type=int, default=int(config['video_length']))
    parser.add_argument('--capture_video', action='store_true')

    # other parameters
    parser.add_argument('--map_index', type=str, default=str(config['map_index']))
    parser.add_argument('--action_queue_size', type=int, default=int(config['action_queue_size']))

    args = parser.parse_args()

    return args


# the choosen maps for experiment
CHOOSEN_MAPS = {
    # 4x4 only blue
    '0': 'maps/4x4/baseOneWorkerMaxResources4x4.xml', # Only blue: 1 base, 1 worker; MaxResources
    # 4x4 not balance
    '1': 'maps/4x4/base4x4.xml', # Blue: 1 base, 1 worker; Red: 1 base; not balance

    # standard
    '2': 'maps/4x4/basesWorkers4x4.xml', # Blue/Red: 1 base, 1 worker
    '3': 'maps/8x8/basesWorkers8x8.xml', # Blue/Red: 1 base, 1 worker
    '4': 'maps/16x16/basesWorkers16x16.xml', # Blue/Red: 1 base, 1 worker
    # standard multi-bases
    '5': 'maps/8x8/TwoBasesWorkers8x8.xml', # Blue/Red: 2 base, 2 worker
    '6': 'maps/8x8/ThreeBasesWorkers8x8.xml',  # Blue/Red: 3 base, 3 worker
    '7': 'maps/8x8/FourBasesWorkers8x8.xml',  # Blue/Red: 4 base, 4 worker
    '8': 'maps/12x12/SixBasesWorkers12x12.xml', # Blue/Red: 6 base, 6 worker
    '9': 'maps/16x16/EightBasesWorkers16x16.xml', # Blue/Red: 8 base, 8 worker
    '10': 'maps/EightBasesWorkers16x12.xml', # Blue/Red: 8 base, 8 worker

    # 8x8 Obstacle
    '11': 'maps/8x8/basesWorkers8x8Obstacle.xml', # Blue/Red: 1 base, 1 worker; Obstacle

    # melee
    '12': 'maps/melee4x4light2.xml',  # Blue/Red: 2 light
    '13': 'maps/melee4x4Mixed2.xml',  # Blue/Red: 1 light, 1 heavy
    '14': 'maps/8x8/melee8x8light4.xml', # Blue/Red: 4 light
    '15': 'maps/8x8/melee8x8Mixed4.xml', # Blue/Red: 2 light, 2 heavy
    '16': 'maps/8x8/melee8x8Mixed6.xml', # Blue/Red: 2 light, 2 heavy, 2 ranged
    '17': 'maps/12x12/melee12x12Mixed12.xml', # Blue/Red: 4 light, 4 heavy, 4 ranged
    '18': 'maps/melee14x12Mixed18.xml',  # Blue/Red: 6 light, 6 heavy, 6 ranged
}



from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
def sample_action(env: MicroRTSGridModeVecEnv):
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
    return action

def en2zh(query: str, from_lang='en', to_lang='zh') -> str:
    import requests
    import random
    import json
    from hashlib import md5
    import os

    # Set your own appid/appkey.
    appid = os.getenv("BAIDU_fANYI_ID")
    appkey = os.getenv("BAIDU_FANYI_KEY")

    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    # from_lang = 'en'
    # to_lang =  'zh'

    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # json.dumps(result, indent=4, ensure_ascii=False)
    result = result['trans_result'][0]['dst']

    return result


def test_en2zh():
    # https://ancient-warriors.fandom.com/wiki/Ranged_Soldier
    sentence = """
A Ranged Soldier is troop that fires projectiles from a distance to eliminate enemy soldiers without putting themselves in danger. These soldiers specialised in slowly weakening Heavy Infantry and killing swarms of Light Infantry with a mass volley of arrows. They lacked in melee combat having to use simple weapons like short swords or daggers they keep on their side. Ranged Soldiers usually lake in armour but make up for it in mobility and can be used effectively alongside Light Soldiers.
"""
    sentence_zh = en2zh(sentence)
    print(sentence_zh)


if __name__ == "__main__":
    test_en2zh()
    pass
