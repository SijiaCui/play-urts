from PLAR.utils.fewshots import *
from PLAR.utils.prompts import *
from PLAR.utils.scripts import *

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

# TASK_SCRIPT_MAPPING = {
#     COA_H_Mineral: harvest_mineral,

#     COA_B_Base: build_base,
#     COA_B_Barrack: build_barrack,

#     COA_P_Worker: produce_worker,
#     COA_P_Light: produce_light_soldier,
#     COA_P_Heavy: produce_heavy_soldier,
#     COA_P_Ranged: produce_ranged_soldier,

#     COA_A_Worker: attack_enemy_worker,
#     COA_A_Buildings: attack_enemy_buildings,
#     COA_A_Soldiers: attack_enemy_soldiers
# }

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


def load_args():
    import json
    import argparse

    parser = argparse.ArgumentParser()

    # load config file to add default arguments
    with open('/root/desc/play-urts/PLAR/configs.json', 'r') as f:
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

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    return args


def gene_attack_param(location, tg_location):
    return (tg_location[0] - location[0] + 3) * 7 + (tg_location[1] - location[1] + 3)

def manhattan_distance(l1, l2) -> int:
    assert len(l1) == len(l2)
    return sum([abs(l1[i] - l2[i]) for i in range(len(l1))])


def get_direction(location, tgt_loc) -> str:
    if tgt_loc[0] > location[0]:
        return "south"
    elif tgt_loc[0] < location[0]:
        return "north"
    elif tgt_loc[1] > location[1]:
        return "east"
    elif tgt_loc[1] < location[1]:
        return "west"
    else:
        return "I only live in two dimensions."


from typing import Tuple, List
class path_planning:
    def __init__(self, valid_map) -> None:
        self.height = len(valid_map)
        self.width = len(valid_map[0])
        self.valid_map = valid_map
        self.n = self.height * self.width

        self.max_dist = 10**10
        # 10**10 is disconnected
        self.shortest_path = [[self.max_dist for _ in range(self.n)] for _ in range(self.n)]
        self.first_step = [[-1 for _ in range(self.n)] for _ in range(self.n)]
        self.run()

        # self.shortest_path = [str(i) for i in self.shortest_path]
        # self.first_step = [str(i) for i in self.first_step]
        # print('\n'.join(self.shortest_path))
        # print('\n'.join(self.first_step))
        # input()

    def run(self):
        # init 
        for i in range(self.n):
            self.shortest_path[i][i] = 0
            self.first_step[i][i] = -1
            self._setup_neighbors(i)

        for k in range(self.n):
            if not self.valid_map[self.index2location(k)]: continue
            for i in range(self.n):
                for j in range(self.n):
                    new_path = self.shortest_path[i][k] + self.shortest_path[k][j]
                    if self.shortest_path[i][j] > new_path:
                        self.shortest_path[i][j] = new_path
                        self.first_step[i][j] = self.first_step[i][k]

    def _setup_neighbors(self, index):
        # north neighbor
        if index // self.width - 1 >= 0:
            self.shortest_path[index][index - self.width] = 1
            self.first_step[index][index - self.width] = DIRECTION_INDEX_MAPPING['north']
        # south neighbor
        if index // self.width + 1 < self.height:
            self.shortest_path[index][index + self.width] = 1
            self.first_step[index][index + self.width] = DIRECTION_INDEX_MAPPING['south']
        
        # east neighbor
        if index % self.width + 1 < self.width:
            self.shortest_path[index][index + 1] = 1
            self.first_step[index][index + 1] = DIRECTION_INDEX_MAPPING['east']
        # west neighbor
        if index % self.width -1 >= 0:
            self.shortest_path[index][index - 1] = 1
            self.first_step[index][index - 1] = DIRECTION_INDEX_MAPPING['west']

    def location2index(self, location: tuple) -> int:
        return location[0] * self.width + location[1]

    def index2location(self, index: int) -> tuple:
        return (index // self.width, index % self.width)

    def get_shortest_path(self, location, tg_location) -> Tuple[int, int]:
        '''
        output: 
            int: path length
            int: direction
        '''
        assert location != tg_location, "Two locations should be different"
        sp = self.shortest_path[self.location2index(location)][self.location2index(tg_location)]
        fs = self.first_step[self.location2index(location)][self.location2index(tg_location)]

        return sp, fs
        if sp < self.max_dist:
            return sp, fs
        else:
            # cant find any path
            return None

    def get_path_nearest(self, location: tuple, targets: List[dict]) -> int:
        min_i = 0
        min_path = self.max_dist
        for i in range(len(targets)):
            cur_path = self.shortest_path[self.location2index(location)][self.location2index(targets[i]['location'])]
            if cur_path < min_path:
                min_i = i
                min_path = cur_path
        return min_i

    def get_manhattan_nearest(self, location: tuple, targets: List[dict]) -> int:
        min_i = 0
        min_dist = self.max_dist
        for i in range(len(targets)):
            cur_dist = manhattan_distance(location, targets[i]['location'])
            if cur_dist < min_dist:
                min_i = i
                min_dist = cur_dist
        return min_i


def go_to(l1, l2) -> str:
    assert l1 != l2
    if l2[0] < l1[0]: return 'north'
    if l2[0] > l1[0]: return 'south'
    if l2[1] < l1[1]: return 'west'
    if l2[1] > l1[1]: return 'east'
    return None


def build_place_invalid(obs, valid_map):
    np.where(obs[:,:,13]==1)
    # TODO 可能存在建房子挡路的情况
    return valid_map
    pass

''' TODO Deprecated Functions

def act_move_autosearch(valid_map: np.ndarray, l1: tuple, l2: tuple) -> Tuple[int, int]:
    """
    Input: valid_map, l1, l2
    Output: 
        int: the length of valid shortest path from l1 to l2
        int: the available direction of the first step
    """
    if l1 == l2:
        return None
    elif distance(l1, l2) == 1:
        return 1, go_to(l1, l2)
    
    # map2d[i][j]==0 -> there is a obstacle
    height = len(valid_map)
    width = len(valid_map[0])

    from queue import Queue
    bfs_queue = Queue(width * height)
    visited = np.zeros(valid_map.shape)
    # location:tuple, distance:int, first_step_direction:str
    r = (l1, 0, None)
    bfs_queue.put(r)
    visited[l1] = 1
    while not bfs_queue.empty():
        item = bfs_queue.get()
        # print(item)
        item_loc = item[0]
        if distance(item_loc, l2) == 1:
            return item[1] + 1, DIRECTION_INDEX_MAPPING[item[2]]
        # 0 <= h < height and 0 <= w < width and map2d[h][w]
        h = item_loc[0] - 1; w = item_loc[1]
        if 0 <= h < height and not visited[h][w] and valid_map[h][w]:
            bfs_queue.put(((h,w), item[1] + 1, item[2] if item[2] else 'north'))
            visited[h][w] = 1
        h = item_loc[0] + 1; w = item_loc[1]
        if 0 <= h < height and not visited[h][w] and valid_map[h][w]:
            bfs_queue.put(((h,w), item[1] + 1, item[2] if item[2] else 'south'))
            visited[h][w] = 1

        h = item_loc[0]; w = item_loc[1] - 1
        if 0 <= w < width and not visited[h][w] and valid_map[h][w]:
            bfs_queue.put(((h,w), item[1] + 1, item[2] if item[2] else 'west'))
            visited[h][w] = 1
        h = item_loc[0]; w = item_loc[1] + 1
        if 0 <= w < width and not visited[h][w] and valid_map[h][w]:
            bfs_queue.put(((h,w), item[1] + 1, item[2] if item[2] else 'east'))
            visited[h][w] = 1
    
    return None
'''

def where_to_build_barrack() -> tuple:
    # TODO 在哪里建造
    return (3,2)


def which_target_to_attack(valid_map: np.ndarray, l1: tuple, target_name='worker', mode=0):
    # TODO 攻击哪个目标
    return obs_json[ENEMY]['worker'][0]


def script_mapping(env, obs: np.ndarray, obs_json: dict) -> np.ndarray:
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from stable_baselines3.common.vec_env import VecVideoRecorder
    assert isinstance(env, MicroRTSGridModeVecEnv) or isinstance(env, VecVideoRecorder)
    assert isinstance(obs_json, dict)

    height = obs_json['env']['height']
    width = obs_json['env']['width']
    action_mask = env.get_action_mask()
    action_mask = action_mask.reshape(-1, action_mask.shape[-1])

    # generate a valid map that indicates that grid is valid to be moved on
    obs = obs.reshape((height, width, -1))
    valid_map = np.zeros(shape=(height, width))
    valid_map[np.where(obs[:,:,13]==1)] = 1 # UNIT_NONE_INDEX
    valid_map = build_place_invalid(obs, valid_map)
    
    path_planer = path_planning(valid_map)
    
    action = np.zeros((len(action_mask), 7), dtype=int)
    # action space: noop/move/harvest/return/produce/attack

    # action for bases: noop, produce
    for i in range(len(obs_json[FIGHT_FOR]['base'])):
        base = obs_json[FIGHT_FOR]['base'][i]
        
        task = base['task']
        location: tuple = base['location']
        index = location[0] * width + location[1]

        # the current action
        if base['action'] == 'produce':
            print(f"Base{str(location)}: current action: {base['action']}")
        elif task == COA_P_Worker:
            # produce worker action
            if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['produce']] == 1:
                directions = np.where(action_mask[index][PRODUCE_DIRECT_1:PRODUCE_DIRECT_2]==1)[0]
                if len(directions) == 0:
                    print(f"Base{str(location)}: can't {task}, no available produce direction")
                    continue

                # action type: produce
                action[index][0] = ACTION_INDEX_MAPPING['produce']
                # produce direction: sample one
                action[index][4] = np.random.choice(directions)
                # action[index][4] = 0 if len(obs_json[FIGHT_FOR]['worker']) == 1 else np.random.choice(directions) # which direction is better? # TODO
                # produce unit type: worker
                action[index][5] = PRODUCE_UNIT_INDEX_MAPPING['worker']
                print(f"Base{str(location)}: {task}")
            else:
                print(f"Base{str(location)}: can't {task}, skipped")
        else:
            # no task
            print(f"Base{str(location)}: do nothing")


    # action for barracks: noop/produce
    for i in range(len(obs_json[FIGHT_FOR]['barrack'])):
        barrack = obs_json[FIGHT_FOR]['barrack'][i]
        
        task = barrack['task']
        location: tuple = barrack['location']
        index = location[0] * width + location[1]

        # the current action
        if barrack['action'] == 'produce':
            print(f"Barrack{str(location)}: current action: {barrack['action']}")
        elif task == COA_P_Light or task == COA_P_Heavy or task == COA_P_Ranged:
            # produce soldiers action
            if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['produce']] == 1:
                directions = np.where(action_mask[index][PRODUCE_DIRECT_1:PRODUCE_DIRECT_2]==1)[0]
                if len(directions) == 0:
                    print(f"Barrack{str(location)}: can't {task}, no available produce direction")
                    continue

                # action type: produce
                action[index][0] = ACTION_INDEX_MAPPING['produce']
                # produce direction: sample one
                action[index][4] = np.random.choice(directions)
                # produce unit type: light/heavy/ranged
                if task == COA_P_Light:
                    action[index][5] = PRODUCE_UNIT_INDEX_MAPPING['light']
                elif task == COA_P_Heavy:
                    action[index][5] = PRODUCE_UNIT_INDEX_MAPPING['heavy']
                else:
                    action[index][5] = PRODUCE_UNIT_INDEX_MAPPING['ranged']
                
                print(f"Barrack{str(location)}: {task}")
            else:
                print(f"Barrack{str(location)}: can't {task}, skipped")
        else:
            # no task
            print(f"Barrack{str(location)}: do nothing")


    # action for workers: noop/move/harvest/return/produce/attack
    for i in range(len(obs_json[FIGHT_FOR]['worker'])):
        worker = obs_json[FIGHT_FOR]['worker'][i]
        
        task = worker['task']
        location: tuple = worker['location']
        index = location[0] * width + location[1]

        # the current action
        if worker['action'] != 'noop':
            print(f"Worker{str(location)}: current action: {worker['action']}")
            continue
        
        random_walk = True  # worker: random walk when doing nothing
        aggressive_units = True   # worker: active attack enemy units
        aggressive_building = True  # worker: active attack enemy buildings
        
        if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['attack']] == 1:
            enemy_units = obs_json[ENEMY]['worker'] + obs_json[ENEMY]['light'] + obs_json[ENEMY]['heavy'] + obs_json[ENEMY]['ranged']
            enemy_buildings = obs_json[ENEMY]['base'] + obs_json[ENEMY]['barrack']

            if aggressive_units and len(enemy_units):
                target = path_planer.get_path_nearest(location, enemy_units)
                tg_name = enemy_units[target]['type']
                tg_location = enemy_units[target]['location']
                if manhattan_distance(location, tg_location) == 1:
                    # the nearest enemy unit in worker's attack range 
                    print(f"Worker{str(location)}: aggressive/attacking enemy {tg_name}{tg_location}")
                    # attack
                    action[index][0] = ACTION_INDEX_MAPPING['attack']
                    action[index][6] = gene_attack_param(location, tg_location)
                    continue
            
            # the nearest enemy unit is not in attack range, but some building is
            if aggressive_building and len(enemy_buildings):
                target = path_planer.get_path_nearest(location, enemy_buildings)
                tg_name = enemy_buildings[target]['type']
                tg_location = enemy_buildings[target]['location']
                if manhattan_distance(location, tg_location) == 1:
                    print(f"Worker{str(location)}: aggressive/attacking enemy {tg_name}{tg_location}")
                    # attack
                    action[index][0] = ACTION_INDEX_MAPPING['attack']
                    action[index][6] = gene_attack_param(location, tg_location)
                    continue
                assert False, "No Way!"

        do_nothing = False
        if task == COA_H_Mineral:
            carrying_resources = worker['resource_num']
            # worker harvest mineral
            if carrying_resources == 0:
                # if not carrying the resource, to harvest
                if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['harvest']] == 1:
                    directions = np.where(action_mask[index][HARVEST_DIRECT_1:HARVEST_DIRECT_2]==1)[0]
                    if len(directions) == 0:
                        print(f"Worker{str(location)}: can't {task}/harvesting, no available harvest direction")
                        continue
                    print(f"Worker{str(location)}: {task}/harvesting")
                    # action type: harvest
                    action[index][0] = ACTION_INDEX_MAPPING['harvest']
                    # harvest direction: sample one
                    action[index][2] = np.random.choice(directions)
                elif action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['move']] == 1:
                    targets = obs_json['env']['resource']
                    if len(targets) == 0:
                        print(f"Worker{str(location)}: can't {task}/moving to mineral, no available mineral")
                        continue
                    # get nearest mineral
                    nm_index = path_planer.get_manhattan_nearest(location, targets)
                    nm_location = targets[nm_index]['location']
                    print(f"Worker{str(location)}: {task}/moving to nearest mineral{nm_location}")
                    # action: move
                    action[index][0] = ACTION_INDEX_MAPPING['move']
                    action[index][1] = path_planer.get_shortest_path(location, nm_location)[1]
                else:
                    do_nothing = True
            else:
                # carrying the resource, to return
                if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['return']] == 1:
                    directions = np.where(action_mask[index][RETURN_DIRECT_1:RETURN_DIRECT_2]==1)[0]
                    if len(directions) == 0:
                        print(f"Worker{str(location)}: can't {task}/returning, no available return direction")
                        continue
                    # action type: return
                    action[index][0] = ACTION_INDEX_MAPPING['return']
                    # return direction: sample one
                    action[index][3] = np.random.choice(directions)
                    print(f"Worker{str(location)}: {task}/returning")
                elif action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['move']] == 1:
                    targets = obs_json[FIGHT_FOR]['base']
                    if len(targets) == 0:
                        print(f"Worker{str(location)}: can't {task}/moving to base, no available base")
                        continue
                    # get nearest base
                    nb_index = path_planer.get_manhattan_nearest(location, targets)
                    nb_location = targets[nb_index]['location']
                    print(f"Worker{str(location)}: {task}/moving to nearest base{nb_location}")
                    # action: move
                    action[index][0] = ACTION_INDEX_MAPPING['move']
                    action[index][1] = path_planer.get_shortest_path(location, nb_location)[1]
                else:
                    do_nothing = True
        elif task == COA_B_Base:
            # worker build base
            # TODO: if exists base, pass
            if len(obs_json[FIGHT_FOR]['base']):
                do_nothing = True
            pass
        elif task == COA_B_Barrack:
            # worker build barrack
            # TODO: if exists barrack, pass
            if len(obs_json[FIGHT_FOR]['barrack']):
                do_nothing = True
            else:
                # build a barrack
                bk_location = where_to_build_barrack()
                if manhattan_distance(location, bk_location) == 1:
                    if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['produce']] == 1:
                        print(f"Worker{str(location)}: {task}/building Barrack{bk_location}")
                        # action: produce barrack
                        action[index][0] = ACTION_INDEX_MAPPING['produce']
                        action[index][4] = DIRECTION_INDEX_MAPPING[go_to(location, bk_location)]
                        action[index][5] = PRODUCE_UNIT_INDEX_MAPPING['barrack']
                    else:
                        # arrival but cant produce
                        do_nothing = True
                elif action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['move']] == 1:
                    print(f"Worker{str(location)}: {task}/moving to building place {bk_location}")
                    # action: move
                    action[index][0] = ACTION_INDEX_MAPPING['move']
                    action[index][1] = path_planer.get_shortest_path(location, bk_location)[1]
        elif task == COA_A_Worker:
            # select target
            if not len(obs_json[ENEMY]['worker']): continue # NO worker to attack
            targets = obs_json[ENEMY]['worker']
            target = path_planer.get_path_nearest(location, targets)
            tg_location = obs_json[ENEMY]['worker'][target]['location']
            shortest_path, direction = path_planer.get_shortest_path(location, tg_location)
            if shortest_path == 1:
                if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['attack']] == 1:
                    print(f"Worker{str(location)}: {task}/attacking enemy Worker{tg_location}")
                    # action: acttack
                    action[index][0] = ACTION_INDEX_MAPPING['attack']
                    action[index][6] = gene_attack_param(location, tg_location)
                else:
                    print(f"Worker{str(location)}: can't {task}/attacking enemy Worker{tg_location}, NOTCLEAR")
            else:
                if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['move']] == 1:
                    print(f"Worker{str(location)}: {task}/closing to enemy Worker{tg_location}")
                    # action: move
                    action[index][0] = ACTION_INDEX_MAPPING['move']
                    action[index][1] = direction
                else:
                    print(f"Worker{str(location)}: can't {task}/closing to enemy Worker{tg_location}")
        elif task == COA_A_Buildings:
            buildings = obs_json[ENEMY]['barrack'] + obs_json[ENEMY]['base']
            # select target
            if not len(buildings): continue # NO Building to attack
            target = path_planer.get_path_nearest(location, buildings)
            tg_name = buildings[target]['type']
            tg_location = buildings[target]['location']
            shortest_path, direction = path_planer.get_shortest_path(location, tg_location)
            if shortest_path == 1:
                if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['attack']] == 1:
                    print(f"Worker{str(location)}: {task}/attacking enemy {tg_name}{tg_location}")
                    # action: attack
                    action[index][0] = ACTION_INDEX_MAPPING['attack']
                    action[index][6] = gene_attack_param(location, tg_location)
                else:
                    print(f"Worker{str(location)}: can't {task}/attacking enemy {tg_name}{tg_location}, NOTCLEAR")
            else:
                if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['move']] == 1:
                    print(f"Worker{str(location)}: {task}/closing to enemy {tg_name}{tg_location}")
                    # action: move
                    action[index][0] = ACTION_INDEX_MAPPING['move']
                    action[index][1] = direction
                else:
                    print(f"Worker{str(location)}: can't {task}/closing to enemy {tg_name}{tg_location}")
        elif task == COA_A_Soldiers:
            # TODO
            pass
        else:
            # no task
            do_nothing = True
        
        if do_nothing:
            if random_walk:
                print(f"Worker{str(location)}: patrolling")
                if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['move']] == 1:
                    directions = np.where(action_mask[index][MOVE_DIRECT_1:MOVE_DIRECT_2]==1)[0]
                    action[index][0] = ACTION_INDEX_MAPPING['move']
                    action[index][1] = np.random.choice(directions)
            else:
                print(f"Worker{str(location)}: do nothing")

    
    # # TODO
    # # from task to action
    # action = sample_action(env)

    return np.array(action)


def sample_action(env):
    import numpy as np
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from stable_baselines3.common.vec_env import VecVideoRecorder
    assert isinstance(env, MicroRTSGridModeVecEnv) or isinstance(env, VecVideoRecorder)

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
    # youdao 
    import uuid
    import requests
    import hashlib
    import time
    import os

    YOUDAO_URL = 'https://openapi.youdao.com/api'
    APP_KEY = os.getenv('YOUDAO_ID')
    APP_SECRET = os.getenv('YOUDAO_KEY')
    
    def truncate(q):
        if q is None:
            return None
        size = len(q)
        return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

    data = {}
    data['from'] = 'EN'
    data['to'] = 'zh-CHS'
    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(query) + salt + curtime + APP_SECRET
    sign = hashlib.sha256(signStr.encode('utf-8')).hexdigest()
    data['appKey'] = APP_KEY
    data['q'] = query
    data['salt'] = salt
    data['sign'] = sign
    # data['vocabId'] = "您的用户词表ID"

    response = requests.post(YOUDAO_URL, data=data).json()
    result = response['translation'][0]
    return result


    # baidu fanyi api
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
