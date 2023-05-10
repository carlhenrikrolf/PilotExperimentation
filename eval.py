import argparse
import matplotlib.pyplot as plt
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

# plot rewards
rewards = pd.read_csv(
        path + 'data.csv',
        index_col='time step',
        usecols=['time step', 'reward'],
    )
rolling_window = int(len(rewards)/20)
plt.plot(rewards.rolling(rolling_window).mean())
plt.xlabel('time step')
plt.ylabel('reward\n(rolling mean over ' + str(rolling_window) + ' time steps)')
plt.savefig(path + 'reward_vs_time.png')
plt.close()

# plot side effects
side_effects_incidence = pd.read_csv(
    path + 'data.csv',
    index_col='time step',
    usecols=['time step', 'side effects incidence'],
)
rolling_window = int(len(side_effects_incidence)/20)
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(side_effects_incidence[:50])
ax1.set_xlabel('time step')
ax1.set_ylabel('side effects incidence')
ax1.set_ylim(-0.03, 1.03)
ax2.plot(side_effects_incidence[50:].rolling(rolling_window).max())
ax2.set_xlabel('time step\n(rolling max over ' + str(rolling_window) + ' time steps)')
ax2.set_ylim(-0.03, 1.03)
fig.savefig(path + 'side_effects_incidence_vs_time.png')

# Calculate time complexity metrics
off_policy_time = pd.read_csv(
    path + 'data.csv',
    index_col='time step',
    usecols=['time step', 'off policy time'],
)
time_complexities = off_policy_time.describe()
time_complexities.to_csv(path + 'time_complexities.csv')