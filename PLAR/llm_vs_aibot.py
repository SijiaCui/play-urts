import sys
import os
import yaml
import numpy as np
from PLAR.grounding import obs_2_text, script_mapping
from PLAR.utils.utils import CHOSEN_MAPS, parse_task, load_args, update_tasks, update_situation, check_task
from PLAR.llm_agents import LLMAgent
from PLAR.utils.metric import Metric
import PLAR.utils as utils
from gym_microrts import microrts_ai
from gym_microrts.envs.plar_vec_env import MicroRTSGridModePLARVecEnv
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
    "naiveMCTSAI": microrts_ai.naiveMCTSAI,
    "mayari": microrts_ai.mayari
}

# ====================
#        Utils
# ====================
def get_run_log_dir(args, engine, map_path):
    """Get the directory to save the run logs."""
    run_dir = f"runs_{engine}_vs_rule/" if args.red in AI_MAPPING else f"runs_rule_vs_{engine}/"
    blue = args.blue if args.blue in AI_MAPPING else args.blue_prompt
    red = args.red if args.red in AI_MAPPING else args.red_prompt
    run_dir += f"{blue}_vs_{red}/"
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
    ai = args.red if args.red in AI_MAPPING else args.blue
    env = MicroRTSGridModePLARVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=args.max_steps,
        ai2s=[AI_MAPPING[ai]],
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
        print(f"Game over at {end_step} step! The winner is {args.blue} with {args.blue_prompt}")
    elif reward[0] < 0:
        print(f"Game over at {end_step} step! The winner is {args.red}")
    else:
        print(f"Game over at {end_step} step! Draw! Between {args.blue} with {args.blue_prompt} and {args.red}")


def switch_fight_for(fight_for):
    if fight_for == "blue":
        utils.FIGHT_FOR = "blue"
        utils.ENEMY = "red"
    else:
        utils.FIGHT_FOR = "red"
        utils.ENEMY = "blue"


def main():
    # ====================
    #        Init
    # ====================
    args = load_args()
    fight_for = "blue" if args.red in AI_MAPPING else "red"
    map_path = CHOSEN_MAPS[args.map_index]
    map_name = map_path.split("/")[-1].split(".xml")[0]

    engine = args.blue if args.red in AI_MAPPING else args.red
    run_dir = get_run_log_dir(args, engine, map_path)
    prompt_config = args.blue_prompt if args.red in AI_MAPPING else args.red_prompt
    llm_agent = LLMAgent(engine, args.temperature, args.max_tokens, map_name, prompt_config)
    env = init_environment(args, map_path, run_dir)
    log_file = init_logging(run_dir)

    obs = env.reset()
    switch_fight_for(fight_for)
    obs_text, obs_dict = obs_2_text(obs[0])
    situation = None
    old_obs = obs_dict
    metric = Metric(obs_dict)

    # ====================
    #        Gaming
    # ====================
    for i in range(args.max_steps):
        print(f"{'-'*20} step-{i} {'-'*20}")

        if i % args.tasks_update_interval == 0:
            response = llm_agent.run(obs_text)
            tasks = parse_task(response)
            situation, _ = update_situation(situation, obs_dict)
            tasks = check_task(tasks, obs_dict, situation)
        else:
            tasks, situation = update_tasks(tasks, situation, obs_dict)
        action_vectors = script_mapping(env, tasks, obs_dict)

        obs, reward, done, info = env.step(action_vectors)
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
