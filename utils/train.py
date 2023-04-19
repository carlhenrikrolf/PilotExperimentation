# import modules
import gymnasium as gym
from json import load, dumps
from numpy import nan
from os import system
import pickle
from pprint import pprint
from sys import stdout

from agents.utils import cellular2tabular

def train(
    agent_id: str,
    config_file_name: str,
    experiment_dir: str,
    render=False,
    quiet=False,
):

    # define output directory
    if experiment_dir[-1] != '/':
        experiment_dir = experiment_dir + '/'
    experiment_path = 'results/' + experiment_dir
    system('mkdir ' + experiment_path)

    # save commits
    system('touch ' + experiment_path + 'commits.txt')
    system('printf "PilotExperimentation\n\n" >> ' + experiment_path + 'commits.txt')
    system('git log -n 1 >> ' + experiment_path + 'commits.txt')
    system('printf "\n\ngym-cellular\n\n" >> ' + experiment_path + 'commits.txt')
    system('cd gym-cellular; git log -n 1 >> ../' + experiment_path + 'commits.txt')

    # load and copy config file
    config_file_path = 'config_files/' + config_file_name
    with open(config_file_path, 'r') as config_file:
        config = load(config_file)
    config['agent'] = agent_id
    with open(experiment_path + config_file_name, 'w') as new_config_file:
        new_config_file.write(dumps(config, indent=4))

    if not quiet:
        print('\nSTARTED TRAINING \n')
        print('configurations:')
        pprint(config)

    # import agent
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
    
    elif 'ucrl2' in agent_id:
        from agents import Ucrl2Agent as Agent
        

    if 'polarisation' in config_file_name:

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
    if not quiet:
        print("state:", state)
        print("time step:", 0, "\n")

    with open(experiment_path + 'data.csv', 'a') as data_file:
        data_file.write('time_step,reward,side_effects_incidence,ns_between_time_steps,ns_between_episodes,explorability')
    with open(experiment_path + 'update_type.csv', 'a') as update_type_file:
        update_type_file.write('time_step')
        for cell in range(0, agt.n_cells):
            update_type_file.write(',new_episode_' + str(cell) + ',new_action_pruning_' + str(cell))

    for time_step in range(config["max_time_steps"]):

        action = agt.sample_action(state)
        R = env.tabular_reward_function(cellular2tabular(env.cellular_encoding(state), agt.n_intracellular_states, agt.n_cells), cellular2tabular(action, agt.n_intracellular_actions, agt.n_cells))
        state, reward, terminated, truncated, info = env.step(action)
        agt.update(state, reward, info["side_effects"])

        with open(experiment_path + 'data.csv', 'a') as data_file:
            data_file.write('\n' + str(time_step + 1) + ',' + str(reward) + ',' + str(env.get_side_effects_incidence()) + ',' + str(agt.get_ns_between_time_steps()) + ',' + str(agt.get_ns_between_episodes()))
            explorability = sum([1 if agt.vk[s,a]+agt.Nk[s,a]>0 else 0 for s in range(agt.n_states) for a in range(agt.n_actions)])
            data_file.write(',' + str(explorability))
        with open(experiment_path + 'update_type.csv', 'a') as update_type_file:
            update_type_file.write('\n' + str(time_step + 1))
            update_type = agt.get_update_type()
            for cell in range(agt.n_cells):
                if cell in update_type['new_episode']:
                    update_type_file.write(',1')
                else:
                    update_type_file.write(',' + str(nan))
                if cell in update_type['new_action_pruning']:
                    update_type_file.write(',1')
                else:
                    update_type_file.write(',' + str(nan))
                

        if terminated or truncated:
            break

        # print
        if not quiet:
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

        # save agent
        if time_step % 10000 == 0:
            with open(experiment_path + '.tmp_agt.pkl', 'wb') as agt_file:
                pickle.dump(agt, agt_file)
            system('cp -f ' + experiment_path + '.tmp_agt.pkl' + ' ' + experiment_path + 'agt.pkl')
            system('rm -f ' + experiment_path + '.tmp_agt.pkl')

    print('\nTRAINING ENDED\n')