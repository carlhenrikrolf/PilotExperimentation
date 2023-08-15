from utils.plot import plot_train_summary, binning, load_and_concatenate, paper_plot, reset_plot, deadlock_plot

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

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
    metavar='<rmax>',
    help='Plot rewards between 0 and rmax',
)
parser.add_argument(
    '--nbins',
    default=50,
    dest='nbins',
    metavar='<nbins>',
    help='Data is averaged over bins of time steps, this is how many such bins there are'
)
parser.add_argument(
    '--style',
    default='paper',
    dest='style',
    metavar='<style>',
    help='This is only used for comparing multiple training runs. By default, style=peper. In addition, style=reset which means that the plot generated is in the style of the figure for the results from the reset environment. If style=deadlock, it will be in the style of the figure for the results from the deadlock environment instead.'
)
parser.add_argument(
    '--zoom',
    default=None,
    dest='zoom',
    metavar='<zoom>',
    help='This is only used for comparing multiple training runs. By default, zoom=-1, which means that the plot generated is not zoomed in. If zoom is set to a positive integer, the plot will be zoomed in on the first zoom time steps.'
)
parser.add_argument(
    '--title',
    default=None,
    dest='title',
    metavar='<title>',
    help='This is only used for comparing multiple training runs. By default, title=None, which means that the plot generated will have no title. If title is set to a string, the plot will have that string as its title.'
)
args = parser.parse_args()

contents = os.listdir('results/' + args.path)

if 'data.csv' in contents:

    # variables
    path = 'results/' + args.path + '/'
    n_bins = args.nbins

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

else:

    # variables
    contents = set(contents)
    contents -= {'metadata.txt'}
    contents -= {'plot.png'}
    path_set = []
    for dir in contents:
        path = 'results/' + args.path + '/' + dir + '/'
        if 'completed.txt' in os.listdir(path):
            path_set.append(path)
        else:
            print('Results in ' + path + ' not complete, skipping...')
    n_bins = args.nbins
    style = args.style
    zoom = args.zoom

    # style
    if style == 'paper':
        if zoom is None:
            zoom = -2
        if args.title is None:
            title = ''
        else:
            title = args.title
        plot = paper_plot
    elif style == 'reset':
        if zoom is None:
            zoom = -2
        if args.title is None:
            title = 'reset variant'
        else:
            title = args.title
        plot = reset_plot
    elif style == 'deadlock':
        if zoom is None:
            zoom = 200
        if args.title is None:
            title = 'deadlock variant'
        else:
            title = args.title
        plot = deadlock_plot
    else:
        raise ValueError('style must be either paper, reset, or deadlock')
    
    # load data
    data, raw_data = load_and_concatenate(path_set, zoom=zoom, n_bins=n_bins)

    # plot data
    fig = plot(data, raw_data, n_bins=n_bins)
    fig.suptitle(title, fontsize=25)
    fig.savefig('results/' + args.path + '/plot.png')
