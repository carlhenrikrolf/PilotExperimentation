import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_train_summary(path, n_bins=50, rmax=None):
    # load data
    first_rows = pd.read_csv(
        path + 'data.csv',
        skiprows=lambda row: row not in [0,1],
    )
    agent = first_rows['agent'][0]
    regulatory_constraints = first_rows['regulatory constraints'][0]
    raw_rewards = pd.read_csv(
        path + 'data.csv',
        index_col='time step',
        usecols=['time step', 'reward'],
    )
    raw_side_effects_incidence = pd.read_csv(
        path + 'data.csv',
        index_col='time step',
        usecols=['time step', 'side effects incidence'],
    )
    if n_bins is not None:
        rewards = binning(raw_rewards, 'reward', n_bins, kind='mean')
        side_effects_incidence = binning(raw_side_effects_incidence, 'side effects incidence', n_bins, kind='max')
    # plot
    summary, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3, 2]})
    text_box = axs[0,1]
    reward_plot = axs[0,0]
    zoom_in = axs[1,1]
    side_effects_plot = axs[1,0]
    text = 'agent:\n'
    text += agent + '\n\n'
    text += 'regulatory constraints:\n'
    text += regulatory_constraints
    text = text.replace(' &', '\n&')
    text_box.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', wrap=True)
    text_box.axis('off')
    if n_bins is not None:
        reward_plot.plot(rewards['time step'].values, rewards['reward'].values)
    else:
        reward_plot.plot(raw_rewards)
    reward_plot.set_xticks([])
    reward_plot.set_ylabel('reward')
    if rmax is not None:
        reward_plot.set_ylim(-0.03, float(rmax)+0.03)
    zoom_in.plot(raw_side_effects_incidence[:50])
    zoom_in.set_xlabel('time step (zoomed in)')
    zoom_in.set_yticks([])
    zoom_in.set_ylim(-0.03, 1.03)
    if n_bins is not None:
        side_effects_plot.plot(side_effects_incidence['time step'].values, side_effects_incidence['side effects incidence'].values)
        side_effects_plot.set_xlabel('time step (' + str(n_bins) + ' bins)')
    else:
        side_effects_plot.plot(raw_side_effects_incidence)
        side_effects_plot.set_xlabel('time step')
    side_effects_plot.set_ylabel('side effects incidence')
    side_effects_plot.set_ylim(-0.03, 1.03)
    return summary


def binning(raw_data, column, n_bins, kind='mean'):
    data = pd.DataFrame(index=range(n_bins+1), columns=['time step', column])
    t = 0
    for bin in range(n_bins):
        next_t = int(np.floor(bin*len(raw_data)/n_bins) + 1)
        data['time step'][bin] = raw_data[column][t:next_t].index.values.max()
        if kind == 'mean':
            data[column][bin] = raw_data[column][t:next_t].values.mean()
        elif kind == 'max':
            data[column][bin] = raw_data[column][t:next_t].values.max()
        else:
            raise ValueError('kind must be either "mean" or "max"')
        t = next_t
    data['time step'][n_bins] = raw_data[column][next_t:].index.values.max()
    if kind == 'mean':
        data[column][n_bins] = raw_data[column][next_t:].values.mean()
    elif kind == 'max':
        data[column][n_bins] = raw_data[column][next_t:].values.max()
    else:
        raise ValueError('kind must be either "mean" or "max"')
    return data