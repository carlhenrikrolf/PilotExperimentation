"""This script produces plots and tables to visualise the data from explore.py"""

# import modules
import matplotlib.pyplot as plt
import numpy as np
from os import system
import pandas as pd
from sys import argv

if len(argv) == 1:

    print('Please provide an argument.')

elif len(argv) == 2:

    experiment_dir = argv[1]

    # define input directory
    if experiment_dir[-1] != '/':
        experiment_dir = experiment_dir + '/'
    if experiment_dir[0] != '.':
        experiment_dir = '.' + experiment_dir
    experiment_path = 'results/' + experiment_dir

    # read data
    rewards = pd.read_csv(
        experiment_path + 'data.csv',
        index_col='time_step',
        usecols=['time_step', 'reward'],
    )

    side_effects_incidence = pd.read_csv(
        experiment_path + 'data.csv',
        index_col='time_step',
        usecols=['time_step', 'side_effects_incidence'],
    )

    ns_between = pd.read_csv(
        experiment_path + 'data.csv',
        index_col='time_step',
        usecols=['time_step', 'ns_between_time_steps', 'ns_between_episodes'],
    )

    # plot data
    plt.plot(rewards)
    plt.xlabel('time step')
    plt.ylabel('reward')
    plt.savefig(experiment_path + 'reward_vs_time.png')
    plt.close()

    plt.plot(side_effects_incidence)
    plt.xlabel('time step')
    plt.ylabel('side effects incidence')
    plt.ylim(-0.03, 1.03)
    plt.savefig(experiment_path + 'side_effects_incidence_vs_time.png')
    plt.close()

    # alternative plots
    plt.plot(rewards.index.values, rewards.values.cumsum() / rewards.index.values)
    plt.xlabel('time step')
    plt.ylabel('cumulative average reward')
    plt.savefig(experiment_path + 'cumulative_average_reward_vs_time.png')
    plt.close()

    plt.plot(side_effects_incidence.cummax())
    plt.xlabel('time step')
    plt.ylabel('cumulative max side effects incidence')
    plt.ylim(-0.03, 1.03)
    plt.savefig(experiment_path + 'cumulative_max_side_effects_incidence_vs_time.png')
    plt.close()

    rolling_window = int(len(rewards)/20)
    plt.plot(rewards.rolling(rolling_window).mean())
    plt.xlabel('time step')
    plt.ylabel('reward\n(rolling mean over ' + str(rolling_window) + ' time steps)')
    plt.savefig(experiment_path + 'rolling_mean_reward_vs_time.png')
    plt.close()

    rolling_window = int(len(side_effects_incidence)/20)
    plt.plot(side_effects_incidence.rolling(rolling_window).max())
    plt.xlabel('time step')
    plt.ylabel('side effects incidence\n(rolling max over ' + str(rolling_window) + ' time steps)')
    plt.ylim(-0.03, 1.03)
    plt.savefig(experiment_path + 'rolling_max_side_effects_incidence_vs_time.png')
    plt.close()

    plt.eventplot(ns_between.index[ns_between['ns_between_episodes'].notnull()])
    plt.xlabel('time step')
    plt.ylabel('episodes')
    plt.yticks([],[])
    plt.savefig(experiment_path + 'episodes.png')
    plt.close()

    # make tables
    s_between = ns_between #/ 1e9
    s_between.rename(
        columns={
            'ns_between_time_steps': 's_between_time_steps',
            'ns_between_episodes': 's_between_episodes',
        },
        inplace=True,
    )
    time_complexities = s_between.describe()
    time_complexities.to_csv(experiment_path + 'time_complexities.csv')

else:
    pass

    # experiment_path = 'results/.'
    # for ind in range(1,len(argv)):
    #     experiment_dir = argv[ind]
    #     if experiment_dir[-1] == '/':
    #         experiment_dir = experiment_dir[:-1]
    #     if experiment_dir[0] == '.':
    #         experiment_dir = experiment_dir[1:]
    #     experiment_path = experiment_path + experiment_dir + '__'
    # experiment_path = experiment_path[:-2] + '/'
    # system('rm -r -f ' + experiment_path + '; mkdir ' + experiment_path) # maybe not remove

    # input_dir_set = set()
    # for arg_ind in range(1,len(argv)):
    #     input_dir = argv[arg_ind]
    #     if input_dir[-1] != '/':
    #         input_dir = input_dir + '/'
    #     if input_dir[0] != '.':
    #         input_dir = '.' + input_dir
    #     input_dir_set += {input_dir}   


    # rewards = [
    #     pd.read_csv(
    #         'results/' + input_dir + 'data.csv',
    #         index_col='time_step',
    #         usecols=['time_step', 'reward'],
    #     ).rename(columns={'reward': input_dir}, inplace=True) for input_dir in input_dir_set
    # ]
    # reward_data = pd.concat(rewards, axis=1, join='inner')
    # plt.plot(reward_data)
    # plt.xlabel('time step')
    # plt.ylabel('reward')
    # plt.legend(reward_data.columns)
    # plt.savefig(experiment_path + 'reward_vs_time.png')