import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

FIGHT_FOR = 'blue'
ENEMY = 'red'

ACTION_TYPE_1 = 0
ACTION_TYPE_2 = 6
MOVE_DIRECT_1 = 6
MOVE_DIRECT_2 = 10
HARVEST_DIRECT_1 = 10
HARVEST_DIRECT_2 = 14
RETURN_DIRECT_1 = 14
RETURN_DIRECT_2 = 18
PRODUCE_DIRECT_1 = 18
PRODUCE_DIRECT_2 = 22
PRODUCE_UNIT_1 = 22
PRODUCE_UNIT_2 = 29
ATTACK_PARAM_1 = 29
ATTACK_PARAM_2 = 78

ACTION_INDEX_MAPPING = {
    'noop': 0,
    'move': 1,
    'harvest': 2,
    'return': 3,
    'produce': 4,
    'attack': 5
}

PRODUCE_UNIT_INDEX_MAPPING = {
    'resource': 0,
    'base': 1,
    'barrack': 2,
    'worker': 3,
    'light': 4,
    'heavy': 5,
    'ranged': 6
}

DIRECTION_INDEX_MAPPING = {
    'north': 0,
    'east': 1,
    'south': 2,
    'west': 3
}

def harvest_mineral(env: MicroRTSGridModeVecEnv, obs_json: dict) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def build_base(env: MicroRTSGridModeVecEnv, obs_json: dict) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def build_barrack(env: MicroRTSGridModeVecEnv, obs_json: dict) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def produce_worker(env: MicroRTSGridModeVecEnv, obs_json: dict) -> list:
    width = obs_json['env']['width']
    height = obs_json['env']['height']
    action_mask = env.get_action_mask()
    action_mask = action_mask.reshape(-1, action_mask.shape[-1])

    action = np.zeros((len(action_mask), 7), dtype=int)
    # 0 action type; 1 move param; 2 harvest param; 3 return param;
    # 4 produce direct; 5 produce unit; 6 attack param 
    for i in range(len(obs_json[FIGHT_FOR]['base'])):
        base = obs_json[FIGHT_FOR]['base'][i]
        index = base['location'][0] * width + base['location'][1]
        
        # if produce is available
        if action_mask[index][ACTION_TYPE_1 + ACTION_INDEX_MAPPING['produce']] == 1:
            # action_i = np.zeros(len(action_mask[index]), dtype=int)

            # # action type: produce
            # action_i[ACTION_TYPE_1 + ACTION_INDEX_MAPPING['produce']] = 1
            # # produce direction: sample one
            # direction = np.random.choice(np.where(action_mask[index][PRODUCE_DIRECT_1:PRODUCE_DIRECT_2]==1)[0]) + PRODUCE_DIRECT_1
            # action_i[direction] = 1
            # # produce unit type: worker
            # action_i[PRODUCE_UNIT_1 + PRODUCE_UNIT_INDEX_MAPPING['worker']] = 1
            # action_i = action_i * action_mask[index]
            # print(action_i)

            # action type: produce
            action[index][0] = ACTION_INDEX_MAPPING['produce']
            # produce direction: sample one
            action[index][4] = np.random.choice(np.where(action_mask[index][PRODUCE_DIRECT_1:PRODUCE_DIRECT_2]==1)[0])
            # produce unit type: worker
            action[index][5] = PRODUCE_UNIT_INDEX_MAPPING['worker']

    action_list = [action]
    return action

def produce_light_soldier(env: MicroRTSGridModeVecEnv, obs_json: dict) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def produce_heavy_soldier(env: MicroRTSGridModeVecEnv, obs_json: dict) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def produce_ranged_soldier(env: MicroRTSGridModeVecEnv, obs_json: dict) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def attack_enemy_worker(env: MicroRTSGridModeVecEnv, obs_json: dict) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def attack_enemy_buildings(env: MicroRTSGridModeVecEnv, obs_json: dict) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def attack_enemy_soldiers(env: MicroRTSGridModeVecEnv, obs_json: dict) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

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


def test_scripts():

    pass


if __name__ == "__main__":
    test_scripts()
    exit()
    
    map_name = 'maps/4x4/base4x4.xml'
    # maps/4x4/baseOneWorkerMaxResources4x4.xml

    # import env and AI bots
    from gym_microrts import microrts_ai
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    from stable_baselines3.common.vec_env import VecVideoRecorder
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
    env.metadata['video.frames_per_second'] = 10

    name_prefix = map_name.split('maps/')[-1].split('.xml')[0].replace('/', '-')
    env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: True, video_length=200, name_prefix=name_prefix)
    obs = env.reset()

    for index in range(200):
        env.render()
        from PLAR.obs2text import obs_2_text
        obs_text, obs_json = obs_2_text(obs)

        # action = sample_action(env)
        action_list = produce_worker(env, obs_json)
        
        for action in action_list:
            obs, reward, done, info = env.step(action)
    
    env.close()

"""
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
"""