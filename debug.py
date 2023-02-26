# import packages
from agents import PeUcrlAgent
from json import load
import gymnasium as gym
from os import system
system("cd ..; pip3 install -e gym-cellular")
import gym_cellular
import numpy as np
from pprint import pprint
from copy import deepcopy

# import configurations
config_file = open("config_files/peucrl_polarisation_2.json", 'r')
config = load(config_file)

print("\nConfig:")
pprint(config)

# instantiate environment
env = gym.make(
    config["environment_version"],
    n_users=config["n_users"],
    n_user_states=config["n_user_states"],
    n_recommendations=config["n_recommendations"],
    n_moderators=config["n_moderators"],
    seed=config["environment_seed"],
)

#def reward_function(x,y):
#    return 0

previous_state, info = env.reset(seed=config["reset_seed"])
print("\n State:")
pprint(previous_state)
print("Info:")
pprint(info)
initial_state = deepcopy(previous_state)

# instantiate agent
agt = PeUcrlAgent(
    confidence_level=config["confidence_level"],
    accuracy=config["accuracy"],
    n_cells=config["n_users"],
    n_intracellular_states=config["n_user_states"] * 2,
    cellular_encoding=env.cellular_encoding,
    n_intracellular_actions=config["n_recommendations"],
    cellular_decoding=env.cellular_decoding,
    reward_function=env.tabular_reward_function,
    cell_classes=config["cell_classes"],
    cell_labelling_function=config["cell_labelling_function"],
    regulatory_constraints=config["constraints_file_path"],
    initial_state=initial_state,
    initial_policy=env.get_initial_policy(),
)



################################################################

# initialise data-saving structures
reward_cumulation = np.zeros(config["max_time_steps"]) * np.nan
side_effects_incidence = np.zeros(config["max_time_steps"]) * np.nan
ns_between_time_steps = np.zeros(config["max_time_steps"]) * np.nan
ns_between_episodes = np.zeros(config["max_time_steps"]) * np.nan

# print
print("")
print("intialisation")
print("state:", previous_state)

for time_step in range(config["max_time_steps"]):

    # interact
    action = agt.sample_action(previous_state)
    current_state, reward, terminated, truncated, info = env.step(action)
    agt.update(current_state, reward, info["side_effects"])

    # save data
    reward_cumulation[time_step] = reward
    side_effects_incidence[time_step] = env.get_side_effects_incidence()
    ns_between_time_steps[time_step] = agt.get_ns_between_time_steps()
    ns_between_episodes[time_step] = agt.get_ns_between_episodes()

    # print
    if time_step <= 1 or terminated or truncated or time_step >= config["max_time_steps"] - 2:
        print("")
        print("Time step:", time_step)
        print("action:", action)
        print("state:", current_state)
        print("reward:", reward)
        print("side effects:")
        pprint(info["side_effects"])

    if terminated or truncated:
        break

    previous_state = current_state

#action = agt.sample_action(previous_state)
#current_state, reward, terminated, truncated, info = env.step(action)
#print("\n Action:")
#pprint(action)
