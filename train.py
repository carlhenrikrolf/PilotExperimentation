"""This script runs an exploring reinforcement learning agent in a specified environments. All specifications for the experiments should be given as a .json file. The path to that file is used as input to this script. Further the name of the output subdirectory in .data/ should be given as the second argument."""

# settings
render = False
debugger = True

# load modules
from utils import train

from os import system
from sys import argv
from time import sleep

system('pip3 install -e gym-cellular -q')

# get arguments
agent_id = argv[1]
config_file_name = argv[2]
experiment_dir = argv[3]

# perform training
if debugger:
    sleep(5) # get time to attach debugger

train(
    agent_id=agent_id,
    config_file_name=config_file_name,
    experiment_dir=experiment_dir,
    render=render,
)


