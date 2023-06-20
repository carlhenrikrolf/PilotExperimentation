# Safe Experimentation in Reinforcement Learning through Pilot Experimentation

This repository is the official implementation of _Safe Experimentation in Reinforcement Learning through Pilot Experimentation_.

## Requirements

There are two git submodules needed, viz., gym-cellular and prism.
The submodule gym_cellular does not need any further installation apart from downloading.
For the submodule prism, it is needed to run makefile to install it.
Follow instructions on their website.
Test that the following command works

```prism -help```

### Python Modules
- argparse
- concurrent
- copy
- cvxpy
- gymnasium
- importlib
- matplotlib
- numpy
- os
- pandas
- pickle
- seaborn
- setuptools
- subprocess
- time
- typing
- tqdm

## Configuration

The directory ```configs``` contains the configuration files for the training runs.
(The subdirectory ```configs\envs``` contains the configurations for the environments.)

## Training

Run the command

```python3 train.py <filename>```

where ```<filename>``` is a configuration file in the directory ```configs```.
Data from the training run is saved in the directory ```results```.

## Evaluation

To plot the results (suggested for longer training runs), run the command

```python3 eval.py <dir>```

where ```<dir>``` a subdirectory in ```results```.

## Results

Discussed results can be seen in ```notebook.ipynb```.

<!-- ## Contributing

Licence -->