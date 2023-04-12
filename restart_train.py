experiment = 'debug'

experiment_path = 'results/' + experiment + '/'
agt_file_path = experiment_path + 'agt.pkl'
config_file_path = experiment_path + 'config.json'

import gymnasium as gym
import json
from os import system
import pickle
from sys import stdout

system('pip3 install -e gym-cellular -q')
import gym_cellular

with open(agt_file_path, 'rb') as agt_file:
    agt = pickle.load(agt_file)
with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

state = agt.observations[0][-1]
last_t = agt.t
env = gym.make(
    config["environment_version"],
    n_users=config["n_users"],
    n_user_states=config["n_user_states"],
    n_recommendations=config["n_recommendations"],
    n_moderators=config["n_moderators"],
    seed=config["environment_seed"],
)

for t in range(last_t, config['max_time_steps']): # redo last time step if incomplete?
    action = agt.sample_action(state)
    state, reward, terminated, truncated, info = env.step(action)
    agt.update(state, reward, info['side_effects'])

    with open(experiment_path + 'data.csv', 'a') as data_file:
        data_file.write('\n' + str(t + 1) + ',' + str(reward) + ',' + str(env.get_side_effects_incidence()) + ',' + str(agt.get_ns_between_time_steps()) + ',' + str(agt.get_ns_between_episodes()))

    # save agent
    if t % 10000 == 0:
        with open(experiment_path + '.tmp_agt.pkl', 'wb') as agt_file:
            pickle.dump(agt, agt_file)
        system('cp -f ' + experiment_path + '.tmp_agt.pkl' + ' ' + experiment_path + 'agt.pkl')
        system('rm -f ' + experiment_path + '.tmp_agt.pkl')

    stdout.write('\033[3K')
    print("time step:", t + 1, end='\r')

    if terminated or truncated:
        break