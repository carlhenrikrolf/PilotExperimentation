from .save import save_data, save_backup

import gymnasium as gym
import os

def instantiate(config, index=None):

    os.system('pip3 install -e gym-cellular -q')
    import gym_cellular
    if index is None:
        env = gym.make(
            *config['env']['args'],
            **config['env']['kwargs'],
        )
        agt = config['agt'](
            seed=config['seed'],
            prior_knowledge=env.prior_knowledge,
            regulatory_constraints=config['regulatory_constraints'],
        )
    else:
        env = gym.make(
            *config['env'][index]['args'],
            **config['env'][index]['kwargs'],
        )
        agt = config['agt'][index](
            seed=config['seed'][index],
            prior_knowledge=env.prior_knowledge,
            regulatory_constraints=config['regulatory_constraints'][index],
        )
    return env, agt
    

def train(
    path: str,
    env,
    agt,
    max_n_time_steps: int,
    restart=False,
    **kwargs,
):

    if restart is True:
        state = env.get_state()
        info = env.get_info()
    else:
        state, info = env.reset()
        agt.reset_seed()
        save_data(path,env=env.get_data(),agt=agt.get_data(),**kwargs)
        

    for t in range(max_n_time_steps):

        action = agt.sample_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        agt.update(state, reward, info)
        save_data(path,env=env.get_data(),agt=agt.get_data(),**kwargs)
        if (t + 1) % 1000 == 0 or t == max_n_time_steps - 1:
            save_backup(path,env,agt)

    with open(path + 'completed.txt', 'a') as completed_file:
        completed_file.write('completed after this number of time steps: ' + str(t + 1) + '\n')




