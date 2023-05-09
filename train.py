# argument
config_module = 'convergence_debug'

# import modules
from utils.train import instantiate, initialize_save, train
from importlib import import_module

# import config
config_module = 'configs.' + config_module
config = import_module(config_module).config

# train
try:
    path = 'results/' + config['super_dir'] + config['dir']
except KeyError:
    path = 'results/' + config['dir']
env, agt = instantiate(config)
max_n_time_steps = config['max_n_time_steps']
initialize_save(path)
train(
    path,
    env,
    agt,
    max_n_time_steps,
)
