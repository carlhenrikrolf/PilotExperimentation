import gymnasium as gym
import os
import pickle as pkl

def instantiate(config):

    os.system('pip3 install -e gym-cellular -q')
    import gym_cellular
    env = gym.make(
        *config['env']['args'],
        **config['env']['kwargs'],
    )
    agt = config['agt'](
        seed=config['seed'],
        prior_knowledge=env.prior_knowledge,
        regulatory_constraints=config['regulatory_constraints'],
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


def initialize_save(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        raise RuntimeError("Data directory '" + path + "' already exists. As a safety mechanism you are required to move it or delete it before running this script again.")
    with open(path + 'data.csv', 'a') as data_file:
        data_file.write('time step,')
        data_file.write('reward,')
        data_file.write('side effects incidence,')
        data_file.write('off policy time,')
        data_file.write('updated cells,')
        data_file.write('update kinds\n')


def save_data(
    path: str,
    time_step: int,
    env: dict,
    agt: dict,
):
            
    with open(path + 'data.csv', 'a') as data_file:
        data_file.write('{:g},'.format(time_step))
        data_file.write(str(env['reward']) + ',')
        data_file.write(str(env['side_effects_incidence']) + ',')
        data_file.write(str(agt['off_policy_time']) + ',')
        data_file.write(str(agt['updated_cells']) + ',')
        data_file.write(str(agt['update_kinds']) + '\n')

def save_backup(path,env,agt):
    with open(path + 'tmp_backup.pkl', 'wb') as backup_file:
        pkl.dump(
            {
                'env': env,
                'agt': agt,
            },
            backup_file,
        )
    os.system('cp -f ' + path + 'tmp_backup.pkl ' + path + 'backup.pkl')
    os.system('rm ' + path + 'tmp_backup.pkl')

