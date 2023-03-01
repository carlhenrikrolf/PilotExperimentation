"""This script produces plots and tables to visualise the data from explore.py"""

# import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sys import argv

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
plt.savefig(experiment_path + 'cumulative_max_side_effects_incidence_vs_time.png')
plt.close()

# make tables
s_between = ns_between / 1e9
s_between.rename(
    columns={
        'ns_between_time_steps': 's_between_time_steps',
        'ns_between_episodes': 's_between_episodes',
    },
    inplace=True,
)
time_complexities = s_between.describe()
time_complexities.to_csv(experiment_path + 'time_complexities.csv')