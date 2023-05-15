import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate results through plotting and calculating metrics')
parser.add_argument(
    'path',
    metavar='<path>',
    help="Evaluates the data in 'results/<path>/'"
)
args = parser.parse_args()

# variables
path = 'results/' + args.path + '/'
n_bins = 50

# plot rewards

def binning(raw_data, column, n_bins):
    data = pd.DataFrame(index=range(n_bins+1), columns=['time step', column])
    t = 0
    for bin in range(n_bins):
        next_t = int(np.floor(bin*len(raw_data)/n_bins) + 1)
        data['time step'][bin] = raw_data[column][t:next_t].index.values.max()
        data[column][bin] = raw_data[column][t:next_t].values.mean()
        t = next_t
    data['time step'][n_bins] = raw_data[column][next_t:].index.values.max()
    data[column][n_bins] = raw_data[column][next_t:].values.mean()
    return data

raw_rewards = pd.read_csv(
    path + 'data.csv',
    index_col='time step',
    usecols=['time step', 'reward'],
)
print('reward data loaded')
rewards = binning(raw_rewards, 'reward', n_bins)
plt.plot(rewards['time step'].values, rewards['reward'].values)
plt.xlabel('time step (' + str(n_bins) + ' bins)')
plt.ylabel('reward')
plt.savefig(path + 'reward_vs_time.png')
plt.close()
print('reward plot saved')

# plot side effects
raw_side_effects_incidence = pd.read_csv(
    path + 'data.csv',
    index_col='time step',
    usecols=['time step', 'side effects incidence'],
)
print('side effects incidence data loaded')
side_effects_incidence = binning(raw_side_effects_incidence, 'side effects incidence', n_bins)
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(raw_side_effects_incidence[:50])
ax1.set_xlabel('time step')
ax1.set_xlim(right=50)
ax1.set_ylabel('side effects incidence')
ax1.set_ylim(-0.03, 1.03)
ax2.plot(side_effects_incidence['time step'].values, side_effects_incidence['side effects incidence'].values)
ax2.set_xlabel('time step (' + str(n_bins) + ' bins)')
ax2.set_xlim(left=50)
ax2.set_ylim(-0.03, 1.03)
fig.savefig(path + 'side_effects_incidence_vs_time.png')
print('side effects incidence plot saved')

# Calculate time complexity metrics
off_policy_time = pd.read_csv(
    path + 'data.csv',
    index_col='time step',
    usecols=['time step', 'off policy time'],
)
print('time complexities data loaded')
time_complexities = off_policy_time.describe()
time_complexities.to_csv(path + 'time_complexities.csv')
print('time complexities saved')