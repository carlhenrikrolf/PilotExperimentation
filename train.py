# argument
config_module = 'simple_ablations'

# import modules
from utils.save import initialize_save, initialize_data
from utils.train import instantiate, train
from importlib import import_module

# import config
config_module = 'configs.' + config_module
config = import_module(config_module).config

# train
path = initialize_save(config)

if type(path) is str:

    env, agt = instantiate(config)
    max_n_time_steps = config['max_n_time_steps']
    initialize_data(path)
    train(path,env,agt,max_n_time_steps)

else: # parallelize

    from concurrent.futures import ProcessPoolExecutor

    n = len(path)
    env = [None] * n
    agt = [None] * n
    max_n_time_steps = [None] * n
    for i in range(n):
        env[i], agt[i] = instantiate(config, index=i)
        max_n_time_steps[i] = config['max_n_time_steps'][i]
        initialize_data(path[i])

    def main():
        with ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
            executor.map(train, path, env, agt, max_n_time_steps)

    if __name__ == '__main__':
        main()
