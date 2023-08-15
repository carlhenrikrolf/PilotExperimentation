# Safe Experimentation in Reinforcement Learning through Pilot Experimentation

This repository is the official implementation of _Safe Experimentation in Reinforcement Learning through Pilot Experimentation_.

## Installation

In this section, we describe how to install the repository on a Linux machine.
We assume that Python 3 and C++ are installed.

First, clone the repository using the `--recursive` option. The option is required to, in addition, clone the two Git submodules `gym-cellular` and `prism`[^clone].

### Prism

The Git submodule `prism` uses Dave Parker's [imc2 branch](https://github.com/davexparker/prism/tree/imc2) of the Prism model-checker.
Note that running the code in this repository requires a version of the Prism model-checker that can verify interval MDPs.
At the time of writing, this is not included in standard installations of the Prism model-checker, and therefore, it is necessary to install it from the submodule.

Before installing the Prism model-checker, it is required to have the Java Development Kit (JDK) installed.
To check whether it is installed use the command ```javac --version```.
(Note that it is not sufficient to merely have the Java Runtime Environment [JRE] installed.)
If it is not installed, it can be downloaded from [Oracle's website](https://jdk.java.net/), and an installation guide can be found [here](https://docs.oracle.com/en/java/javase/20/install/overview-jdk-installation.html).

To install this version of the Prism model-checker, go to the directory `pilot-experimentation/prism/prism`.
Run the command ```make```.
To test the code after installation, run ```make test```.
For additional installation options, consult the [imc2 branch](https://github.com/davexparker/prism/tree/imc2) or [prismmodelchecker.org](https://www.prismmodelchecker.org/manual/InstallingPRISM/Instructions).

### Python Modules
Below, we list the necessary Python modules.
We suggest installing them in a virtual environment[^venv].
- `argparse`
- `gymnasium`
- `importlib`
- `matplotlib`
- `pandas`
- `psutil`
- `seaborn`
- `time`
- `typing`
- `tqdm`
<!-- - cvxpy this one is not in use-->
The modules `numpy` and `setuptools` are also required but are typically installed by default[^defaultModules].
A final Python module requires special attention.
- `gym-cellular`

This Python module is our own and is what is implemented in the Git submodule `gym-cellular`. It can be installed with ```pip3 install -e gym-cellular``` from the `pilot-experimentation` directory.

## Configuration

The directory `configs` contains the configuration files for the training runs.
(The files necessary to reproduce the experiments in _Safe Exploration in Reinforcement Learning through Pilot Experimentation_ are `deadlock.py` and `reset.py`.)
The subdirectory `configs\envs` contains the configurations for the environments.
(The files necessary to reproduce the experiments in _Safe Exploration in Reinforcement Learning through Pilot Experimentation_ are `deadlock_set.py` and `reset_set.py`.)
The seeds can be changed for both the agent and (under `configs\envs`) the environment. If they are set to `None` the seeds are set as a function of the time.


## Training

To start a training run (or set of training runs), from the directory `pilot-experimentation`, use the command

```python3 train.py <filename>```

where `<filename>` is a configuration file in the directory `configs`.
Data from the training run is saved in the directory `results`.
We suggest to detach the terminal while running the training, e.g. by using the command ```screen```.
(To reproduce the experiments from _Safe Exploration in Reinforcement Learning through Pilot Experimentation_, use the commands ```python3 train.py deadlock``` and ```python3 train.py reset``` respectively. Note that these use an upper bound of 50 cores in their parallelisations.)

## Evaluation

To plot the results (suggested for longer training runs), from the directory `pilot-experimentation`, use the command

```python3 eval.py <dir>```

where ```<dir>``` a subdirectory in ```results```.
(To reproduce the plots from _Safe Exploration in Reinforcement Learning through Pilot Experimentation_, use the commands ```python3 eval.py deadlock --style paper --title "deadlock variant"``` and ```python3 eval.py deadlock --style paper --title "reset variant"```  respectively.)

<!-- ## Contributing

Licence -->

## Trouble-Shooting

Error regarding `Process().cpu_num()`. This may occur on some machines, e.g. Mac OS. Edit `agents/peucrl.py`. Replace `cpu_id = Process().cpu_num()` with `cpu_id = 1`. This will not change performance of the algorithm as `cpu_id` is only there for debugging purposes.

## Footnotes

[^clone]: For example, use the command ```git clone --recursive https://github.com/carlhenrikrolf/pilot-experimentation.git```.

[^venv]: A virtual environment can be created with the command ```python -m venv <path/to/new/virtual/environment>```. It can be activated using ```source </path/to/new/virtual/environment>/bin/activate```. Finally, modules can be installed with ```pip3 install <module>```.

[^defaultModules]: Modules such as `concurrent`, `copy`, `os`, `pickle`, `subprocess` are also used but do not need to be installed.