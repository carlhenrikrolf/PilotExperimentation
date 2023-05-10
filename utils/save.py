import copy as cp
import os
import pickle as pkl

def initialize_save(config):
    path = 'results/'
    if 'super_dir' in config:
        path += config['super_dir']
        try:
            assert type(config['dir']) is list
            os.mkdir(path)
            path_list = []
            for dir in config['dir']:
                path_list.append(path + dir)
                os.mkdir(path_list[-1])     
                initialize_data(path_list[-1])
            return path_list
        except FileExistsError:
            raise RuntimeError("Data directory '" + path + "' already exists. As a safety mechanism you are required to move it or delete it before running this script again.")
        except AssertionError:
            raise RuntimeError("There is no need for a 'super_dir' if you only do one experiment.")
    else:
        path += config['dir']
        try:
            os.mkdir(path)
            initialize_data(path)
            return path
        except FileExistsError:
            raise RuntimeError("Data directory '" + path + "' already exists. As a safety mechanism you are required to move it or delete it before running this script again.")
    

def initialize_data(path):
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