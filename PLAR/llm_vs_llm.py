import sys
import os
import yaml
import numpy as np
import PLAR.utils as utils
from PLAR.grounding import obs_2_text, script_mapping
from PLAR.utils.utils import CHOSEN_MAPS, parse_task, load_args, update_tasks, can_we_harvest
from PLAR.llm_agents import LLMAgent
from PLAR.utils.map_info import MAP_INFO
from PLAR.utils.metric import Metric
from gym_microrts.envs.plar_vec_env import MicroRTSGridModePLARVecEnv
from gym_microrts.envs.plar_vec_video_recorder import PLARVecVideoRecorder
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====================
#        Utils
# ====================
def get_run_log_dir(args, map_path):
    """Get the directory to save the run logs."""
    run_dir = "runs/"
    run_dir += f"{args.blue_prompt[0]}-{args.blue}_vs_{args.red_prompt[0]}-{args.red}/"
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
    env = MicroRTSGridModePLARVecEnv(
        num_selfplay_envs=2,
        num_bot_envs=0,
        max_steps=args.max_steps,
        map_paths=[map_path],
        reward_weight=np.array([10, 0, 0, 0, 0, 0]),
        autobuild=False,
    )
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
        print(f"Game over at {end_step} step! The winner is {args.blue} with {args.blue_prompt[0]}-{args.blue_prompt[1]}")
    elif reward[0] < 0:
        print(f"Game over at {end_step} step! The winner is {args.red} with {args.red_prompt[0]}-{args.red_prompt[1]}")
    else:
        print(f"Game over at {end_step} step! Draw! Between {args.blue} with {args.blue_prompt[0]}-{args.blue_prompt[1]} and {args.red} with {args.red_prompt[0]}-{args.red_prompt[1]}")

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
    map_path = CHOSEN_MAPS[str(args.map_index)]
    run_dir = get_run_log_dir(args, map_path)
    map_name = map_path.split("/")[-1].split(".xml")[0]
    llm_agents = [
        LLMAgent(args.blue, args.temperature, args.max_tokens, map_name, args.blue_prompt),
        LLMAgent(args.red, args.temperature, args.max_tokens, map_name, args.red_prompt)
    ]
    log_file = init_logging(run_dir)
    env = init_environment(args, map_path, run_dir)

    obs = env.reset()
    switch_fight_for("blue")
    blue_obs_text, obs_dict = obs_2_text(obs[0])
    switch_fight_for("red")
    red_obs_text, obs_dict = obs_2_text(obs[0])

    situation = MAP_INFO[args.map_index]
    metric = Metric(obs_dict)
    old_obs = obs_dict

    # ====================
    #        Playing
    # ====================
    for i in range(args.max_steps):
        print(f"{'-'*20} step-{i} {'-'*20}")

        if i % args.tasks_update_interval == 0:
            switch_fight_for("blue")
            blue_response = llm_agents[0].run(blue_obs_text)
            switch_fight_for("red")
            red_response = llm_agents[1].run(red_obs_text)

            blue_tasks = parse_task(blue_response)
            red_tasks = parse_task(red_response)

        switch_fight_for("blue")
        blue_tasks, situation = update_tasks(blue_tasks, situation, obs_dict)
        blue_actions = script_mapping(env, blue_tasks, obs_dict)

        switch_fight_for("red")
        red_tasks, situation = update_tasks(red_tasks, situation, obs_dict)
        red_actions = script_mapping(env, red_tasks, obs_dict)

        obs, reward, done, info = env.step(np.array([blue_actions, red_actions]))
        switch_fight_for("blue")
        blue_obs_text, obs_dict = obs_2_text(obs[0])
        switch_fight_for("red")
        red_obs_text, obs_dict = obs_2_text(obs[0])

        metric.update(obs_dict, old_obs)
        old_obs = obs_dict

        if done[0]:
            end_game(env, reward, args, i)
            break

    metric.display(obs_dict)
    log_file.close()


if __name__ == "__main__":
    main()
