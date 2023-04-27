from agents.utils import tabular2cellular

import gymnasium as gym
import json
from os import system
import pickle
from psutil import Process

def restart_train(
    experiment_dir: str,
    n_extra_time_steps: int,
    render=False,
    quiet=True,
):
    
    experiment_path = 'results/' + experiment_dir

    with open(experiment_path + 'agt.pkl', 'rb') as agt_file:
        agt = pickle.load(agt_file)
    with open(experiment_path + 'config.json', 'r') as config_file:
        config = json.load(config_file)

    
    last_t = agt.t
    state = agt.previous_state

    env = gym.make(
        config["environment_version"],
        n_users=config["n_users"],
        n_user_states=config["n_user_states"],
        n_recommendations=config["n_recommendations"],
        n_moderators=config["n_moderators"],
        seed=config["environment_seed"],
    )

    state = tabular2cellular(state, agt.n_states, agt.n_cells)
    state = env.inverse_cellular_encoding(state)

    print('\nTraining Restarted')

    for t in range(last_t, last_t + n_extra_time_steps):

        action = agt.sample_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        agt.update(state, reward, info["side_effects"])

        with open(experiment_path + 'data.csv', 'a') as data_file:
            data_file.write('\n' + str(t + 1) + ',' + str(reward) + ',' + str(env.get_side_effects_incidence()) + ',' + str(agt.get_ns_between_time_steps()) + ',' + str(agt.get_ns_between_episodes()))
            div = (config['n_user_states']*2*config['n_recommendations'])**config['n_users']
            explorability = sum([1 if agt.vk[s,a]+agt.Nk[s,a]>0 else 0 for s in range(agt.n_states) for a in range(agt.n_actions)])/div
            try:
                transfer_explorability = sum([1 if agt.cell_vk[s,a]+agt.cell_Nk[s,a]>0 else 0 for s in range(agt.n_states) for a in range(agt.n_actions)])/div
            except:
                transfer_explorability = explorability
            data_file.write(',{:e}'.format(explorability) + ',{:e}'.format(transfer_explorability))

        if terminated or truncated:
            break

        # save agent
        if t % 10000 == 0:
            with open(experiment_path + '.tmp_agt.pkl', 'wb') as agt_file:
                pickle.dump(agt, agt_file)
            system('cp -f ' + experiment_path + '.tmp_agt.pkl' + ' ' + experiment_path + 'agt.pkl')
            system('rm -f ' + experiment_path + '.tmp_agt.pkl')
    
    print('\nTraining Ended')