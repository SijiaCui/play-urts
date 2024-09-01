import numpy as np
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

def harvest_mineral(env: MicroRTSGridModeVecEnv, obs: np.ndarray) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def build_base(env: MicroRTSGridModeVecEnv, obs: np.ndarray) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def build_barrack(env: MicroRTSGridModeVecEnv, obs: np.ndarray) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def produce_worker(env: MicroRTSGridModeVecEnv, obs: np.ndarray) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def produce_light_soldier(env: MicroRTSGridModeVecEnv, obs: np.ndarray) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def produce_heavy_soldier(env: MicroRTSGridModeVecEnv, obs: np.ndarray) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def produce_ranged_soldier(env: MicroRTSGridModeVecEnv, obs: np.ndarray) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def attack_enemy_worker(env: MicroRTSGridModeVecEnv, obs: np.ndarray) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def attack_enemy_buildings(env: MicroRTSGridModeVecEnv, obs: np.ndarray) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list

def attack_enemy_soldiers(env: MicroRTSGridModeVecEnv, obs: np.ndarray) -> list:
    action_list = []
    action_list.append(sample_action(env))
    return action_list
