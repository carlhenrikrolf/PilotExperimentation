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
    max_n_time_steps: int 
):

    state, info = env.reset()

    for t in range(max_n_time_steps):

        if state==(0,1):
            print('the state!')

        action = agt.sample_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        agt.update(state, reward, info['side_effects'])
        save_data(path,time_step=t+1,env=env.get_data(),agt=agt.get_data())
        if (t + 1) % 1000 == 0 or t == max_n_time_steps - 1:
            save_backup(path,env,agt)




