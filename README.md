# Restricted Boltzmann Machine Flows and The Critical Temperature of Ising models

## Description

Code for *Restricted Boltzmann Machine Flows and The Critical Temperature of Ising models*.

## Prerequisites
- [python](https://www.python.org/) >= 3.6
- [pytorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [numpy](https://www.numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [scipy](https://www.scipy.org/)
- [tensorboardX](https://pypi.org/project/tensorboardX/) 

Typically, [Anaconda](https://www.anaconda.com/distribution/) distribution for Python >= 3.6 is enough. If you choose to use tensorboardX
visualization during Restricted Boltzmann Machine (RBM) training, it is necessary to install it with `pip install tensorboardX`.

## Usage

The main discussion and the flows are presented in the notebooks:

- `main_discussion_L010.ipynb` - Ising square lattice $L\timesL$ wiht $L=10$.
- `main_discussion_L100.ipynb` - Ising square lattice $L\timesL$ wiht $L=100$.

Folders with data and saved trained models:

- `data`: Monte Carlo (MC) samples for the Ising model in a square lattice.
- `NN_trained_models`: trained neural network (NN) thermometers.
- `RBM_trained_models`: trained RBMs.
- `RBM_flows`: RBM flows. 
- `runs`: tensorboardX files if you choose `tensorboard= True` on `rbm.py`.
- `figures`: saved plots.

The classes for MC sampling, NN thermometer and the RBM are presented in the folder `modules`:

- `mc_ising2d.py` 
- `mc_ising2d_MF.py`
- `net.py` 
- `rbm.py` (for GPU computation: `use_cuda= True`)

NN and RBM training in the folders:

- `training_NN_thermometer`
- `training_RBM_thermometer`