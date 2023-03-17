"""This script runs an exploring reinforcement learning agent in a specified environments. All specifications for the experiments should be given as a .json file. The path to that file is used as input to this script. Further the name of the output subdirectory in .data/ should be given as the second argument."""

# settings
render = False
#render = True

# import modules
import gymnasium as gym
from json import load, dumps
from os import system
from pprint import pprint
from sys import argv, stdout
from time import sleep

from agents.utils import cellular2tabular

# take arguments
agent_id = argv[1]
config_file_name = argv[2]
experiment_dir = argv[3]

# define output directory
if experiment_dir[-1] != '/':
    experiment_dir = experiment_dir + '/'
if experiment_dir[0] != '.':
    experiment_dir = '.' + experiment_dir
experiment_path = 'results/' + experiment_dir
system('mkdir ' + experiment_path)

# save commits
system('touch ' + experiment_path + 'commits.txt')
system('printf "PilotExperimentation\n\n" >> ' + experiment_path + 'commits.txt')
system('git log -n 1 >> ' + experiment_path + 'commits.txt')
system('printf "\n\ngym-cellular\n\n" >> ' + experiment_path + 'commits.txt')
system('cd ../gym-cellular; git log -n 1 >> ../PilotExperimentation/' + experiment_path + 'commits.txt')

# load and copy config file
config_file_path = 'config_files/' + config_file_name
config_file = open(config_file_path, 'r')
config = load(config_file)
config_file.close()
config['agent'] = agent_id
system('touch ' + experiment_path + config_file_name)
new_config_file = open(experiment_path + config_file_name, 'w')
new_config_file.write(dumps(config, indent=4))
new_config_file.close()
#system('cp ' + config_file_path + ' ' + experiment_path)
print('\nSTARTED TRAINING \n')
print('configurations:')
pprint(config)

# debugger
sleep(5) # get time to attach debugger

# instantiate polarisation
if 'polarisation' in config_file_name:

    system('cd ..; pip3 install -e gym-cellular -q')
    import gym_cellular

    # instantiate environment
    env = gym.make(
        config["environment_version"],
        n_users=config["n_users"],
        n_user_states=config["n_user_states"],
        n_recommendations=config["n_recommendations"],
        n_moderators=config["n_moderators"],
        seed=config["environment_seed"],
    )


if 'peucrl' in agent_id:

    if agent_id == 'peucrl':
        from agents import PeUcrlAgent as Agent
    elif agent_id == 'peucrl_minus_r':
        from agents import PeUcrlMinusRAgent as Agent
    elif agent_id == 'peucrl_minus_r_minus_shield':
        from agents import PeUcrlMinusRMinusShieldAgent as Agent
    elif agent_id == 'peucrl_minus_r_minus_experimentation':
        from agents import PeUcrlMinusRMinusExperimentationAgent as Agent
    elif agent_id == 'peucrl_minus_r_minus_safety':
        from agents import PeUcrlMinusRMinusSafetyAgent as Agent
    elif agent_id == 'peucrl_minus_evi':
        from agents import PeUcrlMinusEviAgent as Agent
    elif agent_id == 'peucrl_minus_r_minus_evi':
        from agents import PeUcrlMinusRMinusEviAgent as Agent
    elif agent_id == 'peucrl_minus_shield':
        from agents import PeUcrlMinusShieldAgent as Agent
    elif agent_id == 'peucrl_minus_safety':
        from agents import PeUcrlMinusSafetyAgent as Agent
    elif agent_id == 'peucrl_minus_action_pruning':
        from agents import PeUcrlMinusActionPruningAgent as Agent
    elif agent_id == 'peucrl_minus_r_minus_action_pruning':
        from agents import PeUcrlMinusRMinusActionPruningAgent as Agent
    else:
        raise ValueError('Agent not found.')

    # instantiate agent
    agt = Agent(
        confidence_level=config["confidence_level"],
        n_cells=config["n_users"],
        n_intracellular_states=config["n_user_states"] * 2,
        cellular_encoding=env.cellular_encoding,
        n_intracellular_actions=config["n_recommendations"],
        cellular_decoding=env.cellular_decoding,
        reward_function=env.tabular_reward_function,
        cell_classes=env.get_cell_classes(),
        cell_labelling_function=env.get_cell_labelling_function(),
        regulatory_constraints=config["regulatory_constraints"],
        initial_policy=env.get_initial_policy(),
        seed=config["agent_seed"],
    )

# run agent
# initialise environment
state, info = env.reset()

# print
print("state:", state)
print("time step:", 0, "\n")

system('touch ' + experiment_path + 'data.csv')
data_file = open(experiment_path + 'data.csv', 'a')
data_file.write('time_step,reward,side_effects_incidence,ns_between_time_steps,ns_between_episodes')
data_file.close()

# debugger
sleep(5) # for having time to attach

for time_step in range(config["max_time_steps"]):

    action = agt.sample_action(state)
    R = env.tabular_reward_function(cellular2tabular(env.cellular_encoding(state), agt.n_intracellular_states, agt.n_cells), cellular2tabular(action, agt.n_intracellular_actions, agt.n_cells))
    state, reward, terminated, truncated, info = env.step(action)
    agt.update(state, reward, info["side_effects"])

    data_file = open(experiment_path + 'data.csv', 'a')
    data_file.write('\n' + str(time_step + 1) + ',' + str(reward) + ',' + str(env.get_side_effects_incidence()) + ',' + str(agt.get_ns_between_time_steps()) + ',' + str(agt.get_ns_between_episodes()))
    data_file.close()

    if terminated or truncated:
        break

    # print
    if render:
        env.render()
    else:
        if time_step <= 0:
            print("action:", action, "\nstate:", state, "\nreward:", reward, "\nside effects:")
            pprint(info["side_effects"])
            print("time step:", time_step + 1, '\n\n', end='\r')
        elif time_step >= config["max_time_steps"] - 1:
            print("\n\naction:", action, "\nstate:", state, "\nreward:", reward, "\nside effects:")
            pprint(info["side_effects"])
            print("time step:", time_step + 1)
        else:
            stdout.write('\033[3K')
            print("time step:", time_step + 1, end='\r')
    
    assert  R == reward

print('\nTRAINING ENDED\n')