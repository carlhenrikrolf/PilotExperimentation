from agents import PeUcrlAgent
import numpy as np
import gymnasium as gym
import gym_cellular
from copy import deepcopy



confidence_level = 0.95
accuracy = 0.90
n_cells = 2
cell_classes = None
cell_labelling_function = None
regulatory_constraints = None

n_user_states = 2
n_moderators=1
n_recommendations=2

env = gym.make(
    'gym_cellular/Polarisation-v1',
    n_users=n_cells,
    n_user_states=n_user_states,
    n_moderators=n_moderators,
)

def reward_function(state, action):
    return 0

cellular_encoding = env.cellular_encoding
cellular_decoding = env.cellular_decoding

n_intracellular_states = n_user_states * 2
n_intracellular_actions = n_recommendations
initial_policy = np.zeros((n_cells, n_intracellular_states**n_cells), dtype=int) # flat states


agt = PeUcrlAgent(
    confidence_level=confidence_level,
    accuracy=accuracy,
    n_cells=n_cells,
    n_intracellular_states=n_intracellular_states,
    cellular_encoding=cellular_encoding,
    n_intracellular_actions=n_intracellular_actions,
    cellular_decoding=cellular_decoding,
    reward_function=reward_function,
    cell_classes=cell_classes,
    cell_labelling_function=cell_labelling_function,
    regulatory_constraints=regulatory_constraints,
    initial_policy=initial_policy,
)

state, info = env.reset(seed=40)
state = deepcopy(state)
print("state: ", state)
action = agt.sample_action(state)
print("action: ", action)
current_state, reward, terminated, truncated, info = env.step(action)
#current_state = np.zeros(n_cells, dtype=int)
print("st state: ", state)
print("next_state: ", current_state)
#reward = reward_function(state, action)
print("reward: ", reward, env.reward_function(state,action))
side_effects = info['side_effects']
print("side_effects: \n", side_effects)

agt.update(current_state, reward, side_effects=side_effects)

