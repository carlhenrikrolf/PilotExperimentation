# Safe Experimentation in Reinforcement Learning through Pilot Experimentation

This repository is the official implementation of _Safe Experimentation in Reinforcement Learning through Pilot Experimentation_.

See notebook to reproduce analysis.

## Requirements

Python packages:
- numpy
- pandas
- random
- itertools
- gymnasium
- gym_cellular (environments/reinstall_gym_cellular.sh can be run from environments/ if gym_cellular and Pilot Experimentation is in the same directory, i.e., clone them in the same directory)

```pip3 install <requirements>```

Prism model-checker. Follow instructions on their website.

Test that the following command works

```prism -help```

## Exploration

Run the command

```python3 explore.py config_files/<filename>```

where ```<filename>```, for example, is ```peucrl_polarisation_0.json```.

## Evaluation

Run:

```python3 eval.py```

## Results

See this graph

## Contributing

Licence