import ast

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
    import json
    import argparse

    parser = argparse.ArgumentParser()

    # load config file to add default arguments
    with open("/root/desc/play-urts/PLAR/configs.json", "r") as f:
        config = json.load(f)

    # game parameters
    parser.add_argument("--max_steps", type=int, default=int(config["max_steps"]))
    parser.add_argument(
        "--tasks_update_interval",
        type=int,
        default=int(config["tasks_update_interval"]),
    )
    parser.add_argument("--map_index", type=str, default=str(config["map_index"]))

    # llm parameters
    parser.add_argument("--engine", type=str, default=config["llm_engine"])
    parser.add_argument(
        "--temperature", type=float, default=float(config["llm_engine_temperature"])
    )
    parser.add_argument(
        "--max_tokens", type=int, default=int(config["llm_engine_max_tokens"])
    )

    # video recorder parameters
    parser.add_argument("--video_fps", type=int, default=int(config["video_fps"]))
    parser.add_argument("--video_length", type=int, default=int(config["video_length"]))
    parser.add_argument("--capture_video", action="store_true")

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
    else:
        raise ValueError("I only live in two dimensions.")


def find_around_enemies(unit, obs_json):
    enemies = obs_json[ENEMY]
    around_locs = [
        (unit["location"][0], unit["location"][1] - 1),  # north
        (unit["location"][0] - 1, unit["location"][1]),  # east
        (unit["location"][0], unit["location"][1] + 1),  # south
        (unit["location"][0] + 1, unit["location"][1]),  # west
    ]
    attack_enemies = []
    for ENEMY in enemies:
        if tuple(ENEMY["location"]) in around_locs:
            attack_enemies.append(ENEMY)
    return attack_enemies


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
    from PLAR.task2actions import TASK_SPACE

    task_list = []
    params_list = []
    try:
        text = text.split("START of PLAN")[1].split("END of PLAN")[0]
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
            if param_beg + 1 and param_end + 1:
                params = ast.literal_eval(task_with_params[param_beg : param_end + 1])
                task, params = params_valid(task, params)
                if task is not None:
                    task_list.append(task)
                    params_list.append(params)
    except Exception as e:
        print(f"Response Processing Error: {e}")
    print("Parsed Tasks from LLM's Respond:")
    for task, params in zip(task_list, params_list):
        print(task, params)
    return list(zip(task_list, params_list))


def params_valid(task, params):
    from PLAR.task2actions import (
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
