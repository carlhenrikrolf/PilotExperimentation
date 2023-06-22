from utils.plot import plot_train_summary

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
parser.add_argument(
    '--rmax',
    default=None,
    dest='rmax',
    metavar='rmax',
    help='Plot rewards between 0 and rmax',
)
args = parser.parse_args()

# variables
path = 'results/' + args.path + '/'
n_bins = 50

# plot train summary
try:
    fig = plot_train_summary(path, n_bins=n_bins, rmax=args.rmax)
except ValueError:
    fig = plot_train_summary(path, n_bins=None, rmax=args.rmax, plot_side_effects=False)
fig.savefig(path + 'train_summary.png')
print('train summary saved')

# Calculate time complexity metrics
off_policy_time = pd.read_csv(
    path + 'data.csv',
    index_col='time step',
    usecols=['time step', 'off policy time'],
)
time_complexities = off_policy_time.describe()
time_complexities.to_csv(path + 'time_complexities.csv')
print('time complexities saved')