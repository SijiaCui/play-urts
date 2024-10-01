import numpy as np

import json

def check_array(arr: np.ndarray):
    print(f"{'*' * 20}")
    print(f"Array: \n{arr}")
    print(f"Shape: \n{arr.shape}")
    print(f"{'*' * 20}")


def get_json(obs: np.ndarray) -> dict:
    """
    0             5            10       13                 21          27
    [0 1 0 0 0 ## 0 0 0 0 1 ## 1 0 0 ## 0 1 0 0 0 0 0 0 ## 1 0 0 0 0 0] Resource
    [0 1 0 0 0 ## 1 0 0 0 0 ## 0 1 0 ## 0 0 0 0 1 0 0 0 ## 1 0 0 0 0 0] Worker
    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None
    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None

    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None
    [0 0 0 0 1 ## 1 0 0 0 0 ## 0 1 0 ## 0 0 1 0 0 0 0 0 ## 1 0 0 0 0 0] Base
    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None
    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None

    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None
    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None
    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None
    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None

    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None
    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None
    [1 0 0 0 0 ## 1 0 0 0 0 ## 1 0 0 ## 1 0 0 0 0 0 0 0 ## 1 0 0 0 0 0] None
    [0 0 0 0 1 ## 1 0 0 0 0 ## 0 0 1 ## 0 0 1 0 0 0 0 0 ## 1 0 0 0 0 0] Base
    """

    HP_START = 0
    HP_END = 5
    RESOURCE_START = 5
    RESOURCE_END = 10

    OWNER_NONE_INDEX = 10
    OWNER_BLUE_INDEX = 11
    OWNER_RED_INDEX = 12

    UNIT_NONE_INDEX = 13
    UNIT_RESOURCE_INDEX = 14
    UNIT_BASE_INDEX = 15
    UNIT_BARRACK_INDEX = 16
    UNIT_WORKER_INDEX = 17
    UNIT_LIGHT_INDEX = 18
    UNIT_HEAVY_INDEX = 19
    UNIT_RANGED_INDEX = 20

    ACTION_NOOP_INDEX = 21
    ACTION_MOVE_INDEX = 22
    ACTION_HARVEST_INDEX = 23
    ACTION_RETURN_INDEX = 24
    ACTION_PRODUCE_INDEX = 25
    ACTION_ATTACK_INDEX = 26

    ACTION_START = 21
    ACTION_END = 27
    ACTION_MAP = {
        '0': 'noop',
        '1': 'move',
        '2': 'harvest',
        '3': 'return',
        '4': 'produce',
        '5': 'attack'
    }

    data_json = {}

    # 环境
    data_json['env'] = {}
    data_json['env']['height'] = obs.shape[0]
    data_json['env']['width'] = obs.shape[1]
    data_json['blue'] = {}
    data_json['red'] = {}
    data_json['units'] = {}
    for i in range(data_json['env']['height']):
        for j in range(data_json['env']['width']):
            data_json['units'][(i,j)] = {}

    index = np.where((obs[:,:,OWNER_NONE_INDEX]==1) & (obs[:,:,UNIT_RESOURCE_INDEX]==1))
    env_resource_location = np.array(index).T
    env_resource = obs[index]
    # resource_unit_location = np.append(resource_unit_location, [[1,100]], axis=0)

    # check_array(resource_unit_location)
    # check_array(resource_unit_list)

    data_json['env']['resource'] = []
    for i in range(len(env_resource)):
        location = tuple(env_resource_location[i].tolist())
        resource_num = int(np.where(env_resource[i][RESOURCE_START:RESOURCE_END] > 0)[0])
        resource_num = ">= 4" if resource_num == 4 else resource_num
        data_json['env']['resource'].append(
            {
                'owner': 'env',
                'type': 'resource',
                'location': location,
                'resource_num':resource_num
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'env',
            'type': 'resource',
            'location': location,
            'resource_num':resource_num
        }

    # 蓝方情况
    data_json['blue'] = {}
    # 蓝方基地: location, hp, resource_num, action
    index = np.where((obs[:,:,OWNER_BLUE_INDEX]==1) & (obs[:,:,UNIT_BASE_INDEX]==1))
    blue_base = obs[index]
    blue_base_location = np.array(index).T
    data_json['blue']['base'] = []
    for i in range(len(blue_base)):
        location = tuple(blue_base_location[i].tolist())
        hp = int(np.where(blue_base[i][HP_START:HP_END] > 0)[0])
        resource_num = int(np.where(blue_base[i][RESOURCE_START:RESOURCE_END] > 0)[0])
        resource_num = ">= 4" if resource_num == 4 else resource_num
        action = int(np.where(blue_base[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['blue']['base'].append(
            {
                'owner': 'blue',
                'type': 'base',
                'location': location,
                'hp': hp,
                'resource_num': resource_num,
                'action': ACTION_MAP[str(action)],
                'task': '[noop]'
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'blue',
            'type': 'base',
            'location': location,
            'hp': hp,
            'resource_num': resource_num,
            'action': ACTION_MAP[str(action)],
            'task': '[noop]'
        }
    # 蓝方兵营: location, hp, action
    index = np.where((obs[:,:,OWNER_BLUE_INDEX]==1) & obs[:,:,UNIT_BARRACK_INDEX]==1)
    blue_barrack = obs[index]
    blue_barrack_location = np.array(index).T
    data_json['blue']['barrack'] = []
    for i in range(len(blue_barrack)):
        location = tuple(blue_barrack_location[i].tolist())
        hp = int(np.where(blue_barrack[i][HP_START:HP_END] > 0)[0])
        action = int(np.where(blue_barrack[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['blue']['barrack'].append(
            {
                'owner': 'blue',
                'type': 'barrack',
                'location': location,
                'hp': hp,
                'action': ACTION_MAP[str(action)],
                'task': '[noop]'
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'blue',
            'type': 'barrack',
            'location': location,
            'hp': hp,
            'action': ACTION_MAP[str(action)],
            'task': '[noop]'
        }
    # 蓝方工人: location, hp, resource_num, action
    index = np.where((obs[:,:,OWNER_BLUE_INDEX]==1) & obs[:,:,UNIT_WORKER_INDEX]==1)
    blue_worker = obs[index]
    blue_worker_location = np.array(index).T
    data_json['blue']['worker'] = []
    for i in range(len(blue_worker)):
        location = tuple(blue_worker_location[i].tolist())
        hp = int(np.where(blue_worker[i][HP_START:HP_END] > 0)[0])
        resource_num = int(np.where(blue_worker[i][RESOURCE_START:RESOURCE_END] > 0)[0])
        action = int(np.where(blue_worker[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['blue']['worker'].append(
            {
                'owner': 'blue',
                'type': 'worker',
                'location': location,
                'hp': hp,
                'resource_num': resource_num,
                'action': ACTION_MAP[str(action)],
                'task': '[noop]'
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'blue',
            'type': 'worker',
            'location': location,
            'hp': hp,
            'resource_num': resource_num,
            'action': ACTION_MAP[str(action)],
            'task': '[noop]'
        }
    # 蓝方light: location, hp, action
    index = np.where((obs[:,:,OWNER_BLUE_INDEX]==1) & obs[:,:,UNIT_LIGHT_INDEX]==1)
    blue_light = obs[index]
    blue_light_location = np.array(index).T
    data_json['blue']['light'] = []
    for i in range(len(blue_light)):
        location = tuple(blue_light_location[i].tolist())
        hp = int(np.where(blue_light[i][HP_START:HP_END] > 0)[0])
        action = int(np.where(blue_light[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['blue']['light'].append(
            {
                'owner': 'blue',
                'type': 'light',
                'location': location,
                'hp': hp,
                'action': ACTION_MAP[str(action)],
                'task': '[noop]'
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'blue',
            'type': 'light',
            'location': location,
            'hp': hp,
            'action': ACTION_MAP[str(action)],
            'task': '[noop]'
        }
    # 蓝方重型士兵: location, hp, action
    index = np.where((obs[:,:,OWNER_BLUE_INDEX]==1) & obs[:,:,UNIT_HEAVY_INDEX]==1)
    blue_heavy = obs[index]
    blue_heavy_location = np.array(index).T
    data_json['blue']['heavy'] = []
    for i in range(len(blue_heavy)):
        location = tuple(blue_heavy_location[i].tolist())
        hp = int(np.where(blue_heavy[i][HP_START:HP_END] > 0)[0])
        action = int(np.where(blue_heavy[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['blue']['heavy'].append(
            {
                'owner': 'blue',
                'type': 'heavy',
                'location': location,
                'hp': hp,
                'action': ACTION_MAP[str(action)],
                'task': '[noop]'
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'blue',
            'type': 'heavy',
            'location': location,
            'hp': hp,
            'action': ACTION_MAP[str(action)],
            'task': '[noop]'
        }
    # 蓝方远程士兵: location, hp, action
    index = np.where((obs[:,:,OWNER_BLUE_INDEX]==1) & obs[:,:,UNIT_RANGED_INDEX]==1)
    blue_ranged = obs[index]
    blue_ranged_location = np.array(index).T
    data_json['blue']['ranged'] = []
    for i in range(len(blue_ranged)):
        location = tuple(blue_ranged_location[i].tolist())
        hp = int(np.where(blue_ranged[i][HP_START:HP_END] > 0)[0])
        action = int(np.where(blue_ranged[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['blue']['ranged'].append(
            {
                'owner': 'blue',
                'type': 'ranged',
                'location': location,
                'hp': hp,
                'action': ACTION_MAP[str(action)],
                'task': '[noop]'
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'blue',
            'type': 'ranged',
            'location': location,
            'hp': hp,
            'action': ACTION_MAP[str(action)],
            'task': '[noop]'
        }

    # 红方情况
    data_json['red'] = {}
    # 红方基地: location, hp, resource_num, action
    index = np.where((obs[:,:,OWNER_RED_INDEX]==1) & (obs[:,:,UNIT_BASE_INDEX]==1))
    red_base = obs[index]
    red_base_location = np.array(index).T
    data_json['red']['base'] = []
    for i in range(len(red_base)):
        location = tuple(red_base_location[i].tolist())
        hp = int(np.where(red_base[i][HP_START:HP_END] > 0)[0])
        resource_num = int(np.where(red_base[i][RESOURCE_START:RESOURCE_END] > 0)[0])
        resource_num = ">= 4" if resource_num == 4 else resource_num
        action = int(np.where(red_base[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['red']['base'].append(
            {
                'owner': 'red',
                'type': 'base',
                'location': location,
                'hp': hp,
                'resource_num': resource_num,
                'action': ACTION_MAP[str(action)]
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'red',
            'type': 'base',
            'location': location,
            'hp': hp,
            'resource_num': resource_num,
            'action': ACTION_MAP[str(action)]
        }
    # 红方兵营: location, hp, action
    index = np.where((obs[:,:,OWNER_RED_INDEX]==1) & obs[:,:,UNIT_BARRACK_INDEX]==1)
    red_barrack = obs[index]
    red_barrack_location = np.array(index).T
    data_json['red']['barrack'] = []
    for i in range(len(red_barrack)):
        location = tuple(red_barrack_location[i].tolist())
        hp = int(np.where(red_barrack[i][HP_START:HP_END] > 0)[0])
        action = int(np.where(red_barrack[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['red']['barrack'].append(
            {
                'owner': 'red',
                'type': 'barrack',
                'location': location,
                'hp': hp,
                'action': ACTION_MAP[str(action)]
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'red',
            'type': 'barrack',
            'location': location,
            'hp': hp,
            'action': ACTION_MAP[str(action)]
        }
    # 红方工人: location, hp, resource_num, action
    index = np.where((obs[:,:,OWNER_RED_INDEX]==1) & obs[:,:,UNIT_WORKER_INDEX]==1)
    red_worker = obs[index]
    red_worker_location = np.array(index).T
    data_json['red']['worker'] = []
    for i in range(len(red_worker)):
        location = tuple(red_worker_location[i].tolist())
        hp = int(np.where(red_worker[i][HP_START:HP_END] > 0)[0])
        resource_num = int(np.where(red_worker[i][RESOURCE_START:RESOURCE_END] > 0)[0])
        action = int(np.where(red_worker[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['red']['worker'].append(
            {
                'owner': 'red',
                'type': 'worker',
                'location': location,
                'hp': hp,
                'resource_num': resource_num,
                'action': ACTION_MAP[str(action)]
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'red',
            'type': 'worker',
            'location': location,
            'hp': hp,
            'resource_num': resource_num,
            'action': ACTION_MAP[str(action)]
        }
    # 红方light: location, hp, action
    index = np.where((obs[:,:,OWNER_RED_INDEX]==1) & obs[:,:,UNIT_LIGHT_INDEX]==1)
    red_light = obs[index]
    red_light_location = np.array(index).T
    data_json['red']['light'] = []
    for i in range(len(red_light)):
        location = tuple(red_light_location[i].tolist())
        hp = int(np.where(red_light[i][HP_START:HP_END] > 0)[0])
        action = int(np.where(red_light[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['red']['light'].append(
            {
                'owner': 'red',
                'type': 'light',
                'location': location,
                'hp': hp,
                'action': ACTION_MAP[str(action)]
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'red',
            'type': 'light',
            'location': location,
            'hp': hp,
            'action': ACTION_MAP[str(action)]
        }
    # 红方重型士兵: location, hp, action
    index = np.where((obs[:,:,OWNER_RED_INDEX]==1) & obs[:,:,UNIT_HEAVY_INDEX]==1)
    red_heavy = obs[index]
    red_heavy_location = np.array(index).T
    data_json['red']['heavy'] = []
    for i in range(len(red_heavy)):
        location = tuple(red_heavy_location[i].tolist())
        hp = int(np.where(red_heavy[i][HP_START:HP_END] > 0)[0])
        action = int(np.where(red_heavy[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['red']['heavy'].append(
            {
                'owner': 'red',
                'type': 'heavy',
                'location': location,
                'hp': hp,
                'action': ACTION_MAP[str(action)]
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'red',
            'type': 'heavy',
            'location': location,
            'hp': hp,
            'action': ACTION_MAP[str(action)]
        }
    # 红方远程士兵: location, hp, action
    index = np.where((obs[:,:,OWNER_RED_INDEX]==1) & obs[:,:,UNIT_RANGED_INDEX]==1)
    red_ranged = obs[index]
    red_ranged_location = np.array(index).T
    data_json['red']['ranged'] = []
    for i in range(len(red_ranged)):
        location = tuple(red_ranged_location[i].tolist())
        hp = int(np.where(red_ranged[i][HP_START:HP_END] > 0)[0])
        action = int(np.where(red_ranged[i][ACTION_START:ACTION_END] > 0)[0])
        data_json['red']['ranged'].append(
            {
                'owner': 'red',
                'type': 'ranged',
                'location': location,
                'hp': hp,
                'action': ACTION_MAP[str(action)]
            }
        )
        # location index to query unit information
        data_json['units'][location] = {
            'owner': 'red',
            'type': 'ranged',
            'location': location,
            'hp': hp,
            'action': ACTION_MAP[str(action)]
        }

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
def obs_2_text(obs: np.ndarray, zh=False) -> Tuple[str, dict]:
    '''
    Input: 
        np.ndarray: observation

    Output: 
        str: text description of observation
        dict: json type observation
    '''
    # print(f"{'*'*10}Obs2Text: running{'*'*10}", flush=True)
    map_height = obs.shape[1]
    map_width = obs.shape[2]
    text = ''
    obs = obs.reshape((map_height, map_width, -1))
    
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
