# import packages
from agents import PeUcrlAgent
from json import load
import gymnasium as gym
from os import system
system("cd ..; pip3 install -e gym-cellular")
import gym_cellular
import numpy as np
from pprint import pprint

# import configurations
config_file = open("config_files/peucrl_polarisation_1.json", 'r')
config = load(config_file)

# instantiate environment
env = gym.make(
    config["environment_version"],
    n_users=config["n_users"],
    n_user_states=config["n_user_states"],
    n_recommendations=config["n_recommendations"],
    n_moderators=config["n_moderators"],
    seed=config["environment_seed"],
)

def reward_function(x,y):
    return 0

# instantiate agent
agt = PeUcrlAgent(
    confidence_level=config["confidence_level"],
    accuracy=config["accuracy"],
    n_cells=config["n_users"],
    n_intracellular_states=config["n_user_states"] * 2,
    cellular_encoding=env.cellular_encoding,
    n_intracellular_actions=config["n_recommendations"],
    cellular_decoding=env.cellular_decoding,
    reward_function=reward_function,
    cell_classes=config["cell_classes"],
    cell_labelling_function=config["cell_labelling_function"],
    regulatory_constraints=config["regulatory_constraints"],
    initial_policy=env.get_initial_policy(),
)

print("\n Config:")
pprint(config)

################################################################

previous_state, info = env.reset(seed=config["reset_seed"])
print("\n State:")
pprint(previous_state)
print("Info:")
pprint(info)

print(agt.n_intracellular_states, agt.n_states)
print(
    agt._flatten(state=np.array([9,9,9])),
    agt._flatten(state=np.array([0,0,0]))
)
#action = agt.sample_action(previous_state)
#current_state, reward, terminated, truncated, info = env.step(action)
#print("\n Action:")
#pprint(action)
