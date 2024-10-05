import sys
import os
import numpy as np
import PLAR.utils as utils
from PLAR.grounding import obs_2_text, script_mapping
from PLAR.utils.utils import CHOSEN_MAPS, parse_task, load_args, update_tasks
from PLAR.llm_agents import LLMAgent
from PLAR.utils.map_info import MAP_INFO
from PLAR.utils.metric import Metric
from gym_microrts.envs.plar_vec_env import MicroRTSGridModePLARVecEnv
from gym_microrts.envs.plar_vec_video_recorder import PLARVecVideoRecorder


# ====================
#        Utils
# ====================

def init_environment(args, map_name):
    """Initialize the environment and video recorder."""
    env = MicroRTSGridModePLARVecEnv(
        num_selfplay_envs=2,
        num_bot_envs=0,
        max_steps=args.max_steps,
        map_paths=[map_name],
        reward_weight=np.array([10, 0, 0, 0, 0, 0]),
        autobuild=False,
    )
    env.metadata["video.frames_per_second"] = args.video_fps
    name_prefix = map_name.split("maps/")[-1].split(".xml")[0].replace("/", "-")
    env = PLARVecVideoRecorder(
        env,
        f"videos/llm_vs_llm/{args.blue}_vs_{args.red}",
        record_video_trigger=lambda x: True,
        video_length=args.video_length,
        name_prefix=name_prefix,
    )
    return env


def init_logging(args):
    """Initialize logging to a file."""
    os.makedirs("logs/llm_vs_llm", exist_ok=True)
    log_file = open(f"logs/llm_vs_llm/{args.blue}_vs_{args.red}.log", "w")
    sys.stdout = log_file
    return log_file


def end_game(env, reward, args):
    """Handle the end of the game."""
    env.close()
    if reward[0] > 0:
        print(f"Game over! The winner is {args.blue}")
    elif reward[0] < 0:
        print(f"Game over! The winner is {args.red}")
    else:
        print("Game over! Draw!")


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
    llm_agents = [
        LLMAgent(args.blue, args.temperature, args.max_tokens),
        LLMAgent(args.red, args.temperature, args.max_tokens),
    ]
    log_file = init_logging(args)
    map_name = CHOSEN_MAPS[str(args.map_index)]
    env = init_environment(args, map_name)

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
            # blue_response = llm_agents[0].run(blue_obs_text)
            switch_fight_for("blue")
            blue_reponse = """START OF TASK
            [Harvest Mineral](0, 0)
            [Produce Unit]("worker", "south")
            [Produce Unit]("worker", "east")
            [Attack Enemy]("worker", "worker")
            [Attack Enemy]("worker", "worker")
            [Attack Enemy]("worker", "base")
            """
            switch_fight_for("red")
            # red_response = llm_agents[1].run(red_obs_text)
            red_response = """START OF TASK
            [Harvest Mineral](7, 7)
            [Produce Unit]("worker", "west")
            [Produce Unit]("worker", "north")
            [Attack Enemy]("worker", "worker")
            [Attack Enemy]("worker", "worker")
            [Attack Enemy]("worker", "base")
            """

            blue_tasks = parse_task(blue_reponse)
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
            end_game(env, reward, args)
            break

    metric.display(obs_dict)
    log_file.close()


if __name__ == "__main__":
    main()
