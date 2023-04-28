import os

def debug(
    path: str,
    env,
    agt,
    max_n_time_steps: int 
):

    state, info = env.reset()
    save_data(path, initialize=True)

    for t in range(max_n_time_steps):

        action = agt.sample_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        agt.update(state, reward, info['side_effects'])
        save_data(
            path,
            time_step=t,
            env=env.get_data(),
            agt=agt.get_data(),
        )


def save_data(
    path: str,
    initialize: bool = False,
    time_step: int = None,
    env: dict = None,
    agt: dict = None,
):
    
    if initialize:
        try:
            os.mkdir(path)
        except FileExistsError:
            print("Error: Data directory '" + path + "' already exists. As a safety mechanism you are required to move it or delete it before running this script again.")
        with open(path + 'data.csv', 'a') as data_file:
            data_file.write('time step,')
            data_file.write('reward,')
            data_file.write('side effects incidence,')
            data_file.write('off policy time\n')
    else:
        assert time_step is not None and env is not None and agt is not None
        with open(path + 'data.csv', 'a') as data_file:
            data_file.write(str(time_step) + ',')
            data_file.write(str(env['reward']) + ',')
            data_file.write(str(env['side_effects_incidence']) + ',')
            data_file.write(str(agt['off_policy_time']) + '\n')

