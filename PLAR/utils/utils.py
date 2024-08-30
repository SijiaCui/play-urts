from PLAR.utils.fewshots import *
from PLAR.utils.prompts import *

import argparse
import json

def load_args():
    parser = argparse.ArgumentParser()

    # load config file to add default arguments
    with open('configs.json', 'r') as f:
        config = json.load(f)

    parser.add_argument('--engine', type=str, default=config['llm_engine'])
    parser.add_argument('--temperature', type=float, default=float(config['llm_engine_temperature']))
    parser.add_argument('--max_tokens', type=int, default=int(config['llm_engine_max_tokens']))
    
    parser.add_argument('--video_fps', type=int, default=int(config['video_fps']))
    parser.add_argument('--video_length', type=int, default=int(config['video_length']))
    
    parser.add_argument('--capture_video', action='store_true')
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
