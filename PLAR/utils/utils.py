from typing import Tuple, List


FIGHT_FOR = "blue"
ENEMY = "red"

CHOSEN_MAPS = {
    # 4x4 only blue
    "0": "maps/4x4/baseOneWorkerMaxResources4x4.xml",  # Only blue: 1 base, 1 worker; MaxResources
    # 4x4 not balance
    "1": "maps/4x4/base4x4.xml",  # Blue: 1 base, 1 worker; Red: 1 base; not balance
    # standard
    "2": "maps/4x4/basesWorkers4x4.xml",  # Blue/Red: 1 base, 1 worker
    "3": "maps/8x8/basesWorkers8x8.xml",  # Blue/Red: 1 base, 1 worker
    "4": "maps/16x16/basesWorkers16x16.xml",  # Blue/Red: 1 base, 1 worker
    # standard multi-bases
    "5": "maps/8x8/TwoBasesWorkers8x8.xml",  # Blue/Red: 2 base, 2 worker
    "6": "maps/8x8/ThreeBasesWorkers8x8.xml",  # Blue/Red: 3 base, 3 worker
    "7": "maps/8x8/FourBasesWorkers8x8.xml",  # Blue/Red: 4 base, 4 worker
    "8": "maps/12x12/SixBasesWorkers12x12.xml",  # Blue/Red: 6 base, 6 worker
    "9": "maps/16x16/EightBasesWorkers16x16.xml",  # Blue/Red: 8 base, 8 worker
    "10": "maps/EightBasesWorkers16x12.xml",  # Blue/Red: 8 base, 8 worker
    # 8x8 Obstacle
    "11": "maps/8x8/basesWorkers8x8Obstacle.xml",  # Blue/Red: 1 base, 1 worker; Obstacle
    # melee
    "12": "maps/melee4x4light2.xml",  # Blue/Red: 2 light
    "13": "maps/melee4x4Mixed2.xml",  # Blue/Red: 1 light, 1 heavy
    "14": "maps/8x8/melee8x8light4.xml",  # Blue/Red: 4 light
    "15": "maps/8x8/melee8x8Mixed4.xml",  # Blue/Red: 2 light, 2 heavy
    "16": "maps/8x8/melee8x8Mixed6.xml",  # Blue/Red: 2 light, 2 heavy, 2 ranged
    "17": "maps/12x12/melee12x12Mixed12.xml",  # Blue/Red: 4 light, 4 heavy, 4 ranged
    "18": "maps/melee14x12Mixed18.xml",  # Blue/Red: 6 light, 6 heavy, 6 ranged
}

DIRECTION_INDEX_MAPPING = {
    "north": 0,
    "east": 1,
    "south": 2,
    "west": 3,
}

UNIT_DAMAGE_MAPPING = {"worker": 1, "light": 2, "heavy": 4, "ranged": 1}

UNIT_RANGE_MAPPING = {"worker": 1, "light": 1, "heavy": 1, "ranged": 3}

UNIT_HP_MAPPING = {"worker": 1, "light": 4, "heavy": 4, "ranged": 1}

BUILDING_SPACE = ["base", "barrack"]
ALL_UNIT_SPACE = ["worker", "light", "heavy", "ranged", "base", "barrack"]


def load_args():
    import yaml
    import argparse

    parser = argparse.ArgumentParser()

    # load config file to add default arguments
    with open("/root/desc/play-urts/PLAR/configs/configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    # game parameters
    parser.add_argument("--max_steps", type=int, default=int(config["max_steps"]))
    parser.add_argument("--tasks_update_interval", type=int, default=int(config["tasks_update_interval"]))
    parser.add_argument("--map_index", type=str, default=str(config["map_index"]))

    # llm parameters
    parser.add_argument("--blue", type=str, default=config["blue"])
    parser.add_argument("--red", type=str, default=config["red"])
    parser.add_argument("--temperature", type=float, default=float(config["llm_engine_temperature"]))
    parser.add_argument("--max_tokens", type=int, default=int(config["llm_engine_max_tokens"]))
    parser.add_argument("--blue_prompt", nargs='+', type=str, default=config["blue_prompt"])
    parser.add_argument("--red_prompt", nargs='+', type=str, default=config["red_prompt"])

    # video recorder parameters
    parser.add_argument("--video_record", action="store_true")
    parser.add_argument("--video_fps", type=int, default=int(config["video_fps"]))
    parser.add_argument("--video_length", type=int, default=int(config["video_length"]))

    # other parameters
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def manhattan_distance(l1, l2) -> int:
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
    elif tgt_loc == location:
        return "stay"
    else:
        raise ValueError(f"I only live in two dimensions. {location} -> {tgt_loc}")


class path_planning:
    def __init__(self, valid_map) -> None:
        self.height = len(valid_map)
        self.width = len(valid_map[0])
        self.valid_map = valid_map
        self.n = self.height * self.width

        self.max_dist = 10**10
        # 10**10 is disconnected
        self.shortest_path = [
            [self.max_dist for _ in range(self.n)] for _ in range(self.n)
        ]
        self.first_step = [[-1 for _ in range(self.n)] for _ in range(self.n)]
        self.run()

    def run(self):
        # init
        for i in range(self.n):
            self.shortest_path[i][i] = 0
            self.first_step[i][i] = -1
            self._setup_neighbors(i)

        for k in range(self.n):
            if not self.valid_map[self.index2location(k)]:
                continue
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
            self.first_step[index][index - self.width] = DIRECTION_INDEX_MAPPING[
                "north"
            ]
        # south neighbor
        if index // self.width + 1 < self.height:
            self.shortest_path[index][index + self.width] = 1
            self.first_step[index][index + self.width] = DIRECTION_INDEX_MAPPING[
                "south"
            ]
        # east neighbor
        if index % self.width + 1 < self.width:
            self.shortest_path[index][index + 1] = 1
            self.first_step[index][index + 1] = DIRECTION_INDEX_MAPPING["east"]
        # west neighbor
        if index % self.width - 1 >= 0:
            self.shortest_path[index][index - 1] = 1
            self.first_step[index][index - 1] = DIRECTION_INDEX_MAPPING["west"]

    def location2index(self, location: tuple) -> int:
        return location[0] * self.width + location[1]

    def index2location(self, index: int) -> tuple:
        return (index // self.width, index % self.width)

    def get_shortest_path(self, location, tg_location) -> Tuple[int, int]:
        """
        output:
            int: path length
            int: direction
        """
        if location == tg_location:
            return 0, None
        sp = self.shortest_path[self.location2index(location)][
            self.location2index(tg_location)
        ]
        fs = self.first_step[self.location2index(location)][
            self.location2index(tg_location)
        ]

        return sp, fs
        if sp < self.max_dist:
            return sp, fs
        else:
            # cant find any path
            return None

    def get_path_nearest(self, location: tuple, targets: List[tuple]) -> tuple:
        min_i = 0
        min_path = self.max_dist
        for i in range(len(targets)):
            cur_path = self.shortest_path[self.location2index(location)][
                self.location2index(targets[i])
            ]
            if cur_path < min_path:
                min_i = i
                min_path = cur_path
        return targets[min_i]

    def get_manhattan_nearest(self, location: tuple, targets: List[tuple]) -> tuple:
        min_i = 0
        min_dist = self.max_dist
        for i in range(len(targets)):
            cur_dist = manhattan_distance(location, targets[i])
            if cur_dist < min_dist:
                min_i = i
                min_dist = cur_dist
        return targets[min_i]

    def get_locs_with_dist_to_tgt(self, tgt_loc: tuple, dist: int) -> List[tuple]:
        locs = []
        for i in range(self.height):
            for j in range(self.width):
                if manhattan_distance(tgt_loc, (i, j)) <= dist:
                    locs.append((i, j))
        return locs


def parse_task(text: str) -> list:
    import ast
    import re
    from PLAR.grounding import TASK_SPACE

    task_list = []
    params_list = []
    text = text.split("START OF TASK")[1].split("END OF TASK")[0]
    text_list = text.split("\n")
    for task_with_params in text_list:
        task_beg = task_with_params.find("[")
        task_end = task_with_params.find("]")
        param_beg = task_with_params.find("(")
        param_end = task_with_params.rfind(")")
        if (
            task_beg + 1
            and task_end + 1
            and task_with_params[task_beg : task_end + 1] in TASK_SPACE
        ):
            task = task_with_params[task_beg : task_end + 1]
        params = re.sub(r'(?<!\')(\b[a-zA-Z_]+\b)(?!\')', r"'\1'", task_with_params[param_beg : param_end + 1])
        params = re.sub(r"'(\d+)'", r"\1", params)
        if param_beg + 1 and param_end + 1:
            params = ast.literal_eval(params)
            task, params = params_valid(task, params)
            if task is not None:
                task_list.append(task)
                params_list.append(params)
    print("Parsed Tasks from LLM's Respond:")
    for task, params in zip(task_list, params_list):
        print(task, params)
    return list(zip(task_list, params_list))


def parse_tips(text: str) -> str:
    return text.split("START OF TIPS")[1].split("END OF TIPS")[0]


def params_valid(task, params):
    from PLAR.grounding import (
        TASK_DEPLOY_UNIT,
        TASK_HARVEST_MINERAL,
        TASK_BUILD_BUILDING,
        TASK_PRODUCE_UNIT,
        TASK_ATTACK_ENEMY,
    )

    if task == TASK_DEPLOY_UNIT:
        if (
            len(params) == 2
            and params[0] in UNIT_DAMAGE_MAPPING
            and type(params[1]) is tuple
        ):
            return task, params
    if task == TASK_HARVEST_MINERAL:
        if (
            len(params) == 2
            and type(params) is tuple
            and type(params[0]) is int
            and type(params[1]) is int
        ):
            return task, params
    if task == TASK_BUILD_BUILDING:
        if (
            len(params) == 2
            and params[0] in BUILDING_SPACE
            and type(params[1]) is tuple
        ):
            return task, params
    if task == TASK_PRODUCE_UNIT:
        if (
            len(params) == 2
            and params[0] in UNIT_DAMAGE_MAPPING
            and params[1] in DIRECTION_INDEX_MAPPING
        ):
            return task, params
    if task == TASK_ATTACK_ENEMY:
        if (
            len(params) == 2
            and params[0] in UNIT_DAMAGE_MAPPING
            and params[1] in ALL_UNIT_SPACE
        ):
            return task, params

    return None, None


def update_situation(situation, obs_dict):
    import PLAR.utils as utils

    init_situation = {
        "blue": {
            "worker": 0,
            "base": 0,
            "barrack": 0,
            "light": 0,
            "heavy": 0,
            "ranged": 0,
        },
        "red": {
            "worker": 0,
            "base": 0,
            "barrack": 0,
            "light": 0,
            "heavy": 0,
            "ranged": 0,
        },
    }

    if situation is None:
        situation = init_situation

    def update_unit_count(situation_section, obs_section):
        for unit_type in situation_section.keys():
            for unit in obs_section.values():
                if isinstance(unit, dict) and unit["type"] == unit_type:
                    situation_section[unit_type] += 1

    new_situation = {}
    new_situation[utils.FIGHT_FOR] = {unit_type: 0 for unit_type in situation[utils.FIGHT_FOR].keys()}
    new_situation[utils.ENEMY] = {unit_type: 0 for unit_type in situation[utils.ENEMY].keys()}

    update_unit_count(new_situation[utils.FIGHT_FOR], obs_dict[utils.FIGHT_FOR])
    update_unit_count(new_situation[utils.ENEMY], obs_dict[utils.ENEMY])

    return new_situation, situation


def update_tasks(tasks: List[Tuple], situation, obs_dict):
    import PLAR.utils as utils
    from PLAR.grounding import TASK_BUILD_BUILDING, TASK_PRODUCE_UNIT, TASK_ATTACK_ENEMY

    new_situation, old_situation = update_situation(situation, obs_dict)

    def process_tasks(tasks, situation_key, unit_index, task_types, changes_condition):
        for unit_type in new_situation[situation_key].keys():
            changes = max(changes_condition(unit_type), 0)
            for task in tasks:
                if (task[0] in task_types and task[1][unit_index] == unit_type and changes > 0):
                    print(f"Completed task: {task[0]}{task[1]}")
                    tasks.remove(task)
                    changes -= 1
                    if changes == 0:
                        break

    process_tasks(tasks, utils.FIGHT_FOR, 0, [TASK_PRODUCE_UNIT, TASK_BUILD_BUILDING],
        lambda unit_type: new_situation[utils.FIGHT_FOR][unit_type] - old_situation[utils.FIGHT_FOR][unit_type])

    process_tasks(tasks, utils.ENEMY, 1, [TASK_ATTACK_ENEMY],
        lambda unit_type: old_situation[utils.ENEMY][unit_type] - new_situation[utils.ENEMY][unit_type])
    tasks = can_we_harvest(tasks, obs_dict, situation)
    return tasks, new_situation

def can_we_harvest(tasks, obs_dict, situation):
    import PLAR.utils as utils
    from PLAR.grounding import TASK_HARVEST_MINERAL

    num_worker_with_resource = 0
    for unit in obs_dict[utils.FIGHT_FOR].values():
        if isinstance(unit, dict) and unit["type"] == "worker":
            if unit["resource_num"] > 0:
                num_worker_with_resource += 1

    for task in tasks:
        if task[0] == TASK_HARVEST_MINERAL:
            is_exist_mine = obs_dict["units"][task[1]] and obs_dict["units"][task[1]]["type"] == "resource"
            is_exist_base = situation[utils.FIGHT_FOR]["base"] > 0

            if is_exist_base and (num_worker_with_resource > 0 or is_exist_mine):
                num_worker_with_resource -= 1
            else:
                tasks.remove(task)
    return tasks

if __name__ == '__main__':
    aa=0
    if aa:
        print("aa")
    else:
        print("bb")
