import numpy as np
from typing import Tuple
import PLAR.utils.utils as utils


def get_json(obs: Tuple[np.ndarray, np.ndarray]) -> dict:

    UNIT_ID_INDEX = 0
    HP_INDEX = 1
    RESOURCE_INDEX = 2
    OWNER_INDEX = 3
    UNIT_TYPE_INDEX = 4
    CUR_ACTION_INDEX = 5

    ACTION_MAP = {
        "0": "noop",
        "1": "move",
        "2": "harvest",
        "3": "return",
        "4": "produce",
        "5": "attack",
    }

    UNIT_TYPE_MAP = {
        "1": "resource",
        "2": "base",
        "3": "barrack",
        "4": "worker",
        "5": "light",
        "6": "heavy",
        "7": "ranged",
    }

    OWNER_MAP = {"blue": 1, "red": 2}

    obs_json = {}

    # 环境
    obs, resources = obs[0], obs[1]
    obs = np.squeeze(obs)
    resources = np.squeeze(resources)
    obs_json["env"] = {}
    obs_json["env"]["height"] = obs.shape[1]
    obs_json["env"]["width"] = obs.shape[2]
    obs_json["units"] = {}
    for i in range(obs_json["env"]["height"]):
        for j in range(obs_json["env"]["width"]):
            obs_json["units"][(i, j)] = {}

    # env
    locations = np.where(obs[UNIT_TYPE_INDEX] == 1)
    locations = np.array(locations).T
    obs_json["env"]["resource"] = []
    for location in locations:
        location = tuple(location.tolist())
        d = {
            "owner": "env",
            "type": "resource",
            "location": location,
            "resource_num": int(obs[RESOURCE_INDEX][location]),
            "id": int(obs[UNIT_ID_INDEX][location]),
        }
        obs_json["env"]["resource"].append(d)
        obs_json["units"][location] = d

    # units
    for owner in OWNER_MAP.keys():
        obs_json[owner] = {}
        obs_json[owner]["resources"] = resources[OWNER_MAP[owner] - 1]
        locations = np.where(obs[OWNER_INDEX] == OWNER_MAP[owner])
        locations = np.array(locations).T
        for location in locations:
            location = tuple(location.tolist())
            unit_id = int(obs[UNIT_ID_INDEX][location])
            d = {
                "owner": owner,
                "type": UNIT_TYPE_MAP[str(obs[UNIT_TYPE_INDEX][location])],
                "location": location,
                "hp": int(obs[HP_INDEX][location]),
                "action": ACTION_MAP[str(obs[CUR_ACTION_INDEX][location])],
                "id": unit_id,
                "resource_num": int(obs[RESOURCE_INDEX][location]),
                "task_type": "[noop]",
                "task_params": (),
            }
            obs_json[owner][unit_id] = d
            obs_json["units"][location] = d
    return obs_json


CONST_FIGHT_FOR = "blue team"
CONST_ENEMY = "red team"

CONST_MINERAL = "Mineral Fields"
CONST_BASE = "base"
CONST_BARRACK = "barrack"
CONST_WORKER = "worker"
CONST_LIGHT = "light soldier"
CONST_HEAVY = "heavy soldier"
CONST_RANGED = "ranged soldier"


def get_env_text(env_data: dict) -> str:
    text = f"The Game map is a {env_data['height']}x{env_data['width']} grid.\n"
    resources = env_data["resource"]
    text += f"Available {CONST_MINERAL}: {len(resources)}\n"
    for resource in resources:
        text += f"- {CONST_MINERAL}{resource['location']} resource: {resource['resource_num']}\n"
    text += f"You are in the {utils.FIGHT_FOR} side.\n"
    return text


def get_player_text(player_data: dict, player: str) -> str:
    data = {}
    data["base"] = []
    data["barrack"] = []
    data["worker"] = []
    data["light"] = []
    data["heavy"] = []
    data["ranged"] = []
    for unit in player_data.values():
        if isinstance(unit, dict):
            data[unit["type"]].append(unit)

    text = f"{player.capitalize()}'s Units:\n"
    for unit_type, units in data.items():
        text += f"{unit_type}: {len(units)}\n"
        for unit in units:
            if player == utils.FIGHT_FOR:
                text += f"- {unit['location']}, task: {unit['task_type']}, action: {unit['action']}\n"
            else:
                text += f"- {unit['location']}, action: {unit['action']}\n"
    return text


def obs_2_text(obs: Tuple[np.ndarray, np.ndarray]) -> Tuple[str, dict]:
    data = get_json(obs)
    text_env = get_env_text(data["env"])
    text_ENEMY = get_player_text(data[utils.ENEMY], utils.ENEMY)
    text_FIGHT_FOR = get_player_text(data[utils.FIGHT_FOR], utils.FIGHT_FOR)
    return text_env + "\n" + text_ENEMY + "\n" + text_FIGHT_FOR, data


def show_all_maps_figure():
    """Visualize all maps, located in `show_maps` directory."""

    import os

    # import env and AI bots
    from gym_microrts import microrts_ai
    from gym_microrts.envs.plar_vec_env import MicroRTSGridModePLARVecEnv
    from gym_microrts.envs.plar_vec_video_recorder import PLARVecVideoRecorder

    map_dir = "/root/desc/play-urts/gym_microrts/microrts/maps"

    all_map_list = []
    for path, __, files in os.walk(map_dir):
        for file_name in files:
            all_map_list.append(os.path.join(path, file_name))
    all_map_list = sorted(all_map_list)

    all_map_list = all_map_list[1:]  # remove /maps/.DS_Store

    print(",\n".join(all_map_list))

    for tt in all_map_list:
        map_name = tt[tt.find("maps/") :]
        envs = MicroRTSGridModePLARVecEnv(
            num_selfplay_envs=0,
            num_bot_envs=1,
            max_steps=2000,
            render_theme=2,
            ai2s=[microrts_ai.coacAI for _ in range(1)],
            map_paths=[map_name],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            # autobuild=False
        )
        name_prefix = map_name.split("maps/")[-1].split(".xml")[0].replace("/", "-")

        envs = PLARVecVideoRecorder(
            envs,
            "show_maps",
            record_video_trigger=lambda x: x == 0,
            video_length=1,
            name_prefix=name_prefix,
        )
        obs = envs.reset()
        envs.close()

    os.system("rm /root/desc/play-urts/PLAR/show_maps/*.json")


def test_obs_2_text():
    import os

    # import env and AI bots
    from gym_microrts import microrts_ai
    from gym_microrts.envs.plar_vec_env import MicroRTSGridModePLARVecEnv

    from PLAR.utils.utils import CHOSEN_MAPS

    print(len(CHOSEN_MAPS), CHOSEN_MAPS)

    for map_name in CHOSEN_MAPS.values():
        envs = MicroRTSGridModePLARVecEnv(
            num_selfplay_envs=0,
            num_bot_envs=1,
            max_steps=2000,
            render_theme=2,
            ai2s=[microrts_ai.coacAI for _ in range(1)],
            map_paths=[map_name],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            autobuild=False,
        )
        name_prefix = map_name.split("maps/")[-1].split(".xml")[0].replace("/", "-")

        obs = envs.reset()

        obs_text, obs_json = obs_2_text(obs)
        path = "texts/"
        os.makedirs(path, exist_ok=True)
        with open(path + name_prefix + ".txt", "w") as f:
            f.write(str(obs_json) + "\n\n")
            f.write(obs_text + "\n")

        envs.close()


if __name__ == "__main__":
    # show_all_maps_figure()
    test_obs_2_text()
    exit()
