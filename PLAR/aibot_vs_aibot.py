import sys
import os
import yaml
import numpy as np
from PLAR.utils.utils import load_args, CHOSEN_MAPS
from PLAR.grounding import obs_2_text
from PLAR.utils.metric import Metric
from gym_microrts import microrts_ai
from gym_microrts.envs.plar_vec_env import MicroRTSBotPLARVecEnv
from gym_microrts.envs.plar_vec_video_recorder import PLARVecVideoRecorder
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

AI_MAPPING = {
    "randomBiasedAI": microrts_ai.randomBiasedAI,
    "randomAI": microrts_ai.randomAI,
    "passiveAI": microrts_ai.passiveAI,
    "workerRushAI": microrts_ai.workerRushAI,
    "lightRushAI": microrts_ai.lightRushAI,
    "coacAI": microrts_ai.coacAI,
    "naiveMCTSAI": microrts_ai.naiveMCTSAI
}


# ====================
#        Utils
# ====================
def get_run_log_dir(args, map_path):
    """Get the directory to save the run logs."""
    run_dir = "runs/rule_vs_rule/"
    run_dir += f"{args.blue}_vs_{args.red}/"
    run_dir += map_path.split("maps/")[-1].split(".xml")[0].replace("/", "-")
    for i in range(1, int(1e9)):
        if not os.path.exists(f"{run_dir}_{i}"):
            run_dir += f"_{i}/"
            os.makedirs(run_dir, exist_ok=True)
            break
    args_dict = vars(args)
    with open(f"{run_dir}/configs.yaml", "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False, allow_unicode=True)
    return run_dir


def init_environment(args, map_path, run_dir):
    """Initialize the environment and video recorder."""
    env = MicroRTSBotPLARVecEnv(
        ai1s=[AI_MAPPING[args.blue]],
        ai2s=[AI_MAPPING[args.red]],
        max_steps=args.max_steps,
        map_paths=[map_path],
        reward_weight=np.array([10, 0, 0, 0, 0, 0]),
        autobuild=False,
    )
    if args.record_video:
        env.metadata["video.frames_per_second"] = args.video_fps
        env = PLARVecVideoRecorder(
            env,
            run_dir,
            record_video_trigger=lambda x: True,
            video_length=args.video_length,
        )
    return env


def init_logging(run_dir):
    """Initialize logging to a file."""
    log_file = open(run_dir + "run.log", "w")
    sys.stdout = log_file
    return log_file


def end_game(env, reward, args, end_step):
    """Handle the end of the game."""
    env.close()
    print("\n")
    if reward[0] > 0:
        print(f"Game over at {end_step} step! The winner is blue {args.blue}")
    elif reward[0] < 0:
        print(f"Game over at {end_step} step! The winner is red {args.red}")
    else:
        print(f"Game over at {end_step} step! Draw! Between {args.blue} and {args.red}")


def main():
    # ====================
    #        Init
    # ====================
    args = load_args()
    map_path = CHOSEN_MAPS[args.map_index]
    run_dir = get_run_log_dir(args, map_path)
    env = init_environment(args, map_path, run_dir)
    log_file = init_logging(run_dir)

    obs = env.reset()
    obs_text, obs_dict = obs_2_text(obs[0])
    old_obs = obs_dict
    metric = Metric(obs_dict)

    # ====================
    #        Gaming
    # ====================
    for i in range(args.max_steps):
        # FIXME: Could we do without this useless action?
        actions = np.zeros((obs_dict["env"]["height"] * obs_dict["env"]["width"], 7))
        obs, reward, done, info = env.step(actions)  # ai bot will ignore these actions
        obs_text, obs_dict = obs_2_text(obs[0])
        metric.update(obs_dict, old_obs)
        old_obs = obs_dict

        if done[0]:
            end_game(env, reward, args, i)
            break

    metric.display(obs_dict)
    log_file.close()

if __name__ == "__main__":
    main()
