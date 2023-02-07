from agents import PeUcrlAgent
import numpy as np

confidence_level = 0.95
accuracy = 0.90
n_cells = 2
n_intracellular_actions = 2
def cellular_encoding(state):
    return state
n_intracellular_states = 2
def cellular_decoding(action):
    return action
def reward_function(state, action):
    return 0
cell_classes = None
cell_labelling_function = None
regulatory_constraints = None
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

state = np.zeros(n_cells, dtype=int)
print("state: ", state)
action = agt.sample_action(state)
print("action: ", action)
current_state = np.zeros(n_cells, dtype=int)
print("next_state: ", current_state)
reward = reward_function(state, action)
print("reward: ", reward)
side_effects = np.array(
    [
        ['safe', 'silent'],
        ['silent', 'unsafe'],
    ]
)
print("side_effects: \n", side_effects)

agt.update(current_state, reward, side_effects=side_effects)

