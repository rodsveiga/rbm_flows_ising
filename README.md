# Restricted Boltzmann Machine Flows and The Critical Temperature of Ising models

## Description

Code for *Restricted Boltzmann Machine Flows and The Critical Temperature of Ising models* (Under review).

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

- `main_discussion_L010.ipynb`.
- `main_discussion_L100.ipynb`.

Further analysis about learned weight matrices:

- `weight_analysis_L010.ipynb`.
- `weight_analysis_L036.ipynb`.
- `weight_analysis_L048.ipynb`.
- `weight_analysis_L100.ipynb`.
- `weight_analysis_comparing_L.ipynb`.


Folders with data and saved trained models:

- `data`: Monte Carlo (MC) samples for the Ising model in a square lattice.
- `NN_trained_models`: trained neural network (NN) thermometers.
- `RBM_trained_models`: trained RBMs.
- `RBM_flows`: RBM flows.
- `runs`: tensorboardX files if you choose `tensorboard=True` on `rbm.py`.
- `figures`: saved plots.
- `weight_analysis`: singular value and eigenvalue decompositions.

The classes for the MC sampling, the NN thermometer and the RBM are presented in the folder `modules`:

- `mc_ising2d.py` 
- `mc_ising2d_MF.py`
- `net.py` 
- `rbm.py` (for GPU computation: `use_cuda=True`)

NN and RBM training in the folders:

- `training_NN_thermometer`
- `training_RBM`

## License

See [LICENSE](https://github.com/rodsveiga/rbm_flows_ising/blob/master/LICENSE).

## References

- Scale-invariant feature extraction of neural network and renormalization group flow, [Phys. Rev. E 97, 053304 (2018)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.97.053304).
- An exact mapping between the Variational Renormalization Group and Deep Learning, [arXiv:1410.3831](https://arxiv.org/abs/1410.3831) (2014).
- Deep learning and the renormalization group, [arXiv:1301.3124](https://arxiv.org/abs/1301.3124) (2013).
- A high-bias, low-variance introduction to machine learning for physicists, Physics Reports, [https://doi.org/10.1016/j.physrep.2019.03.001](https://doi.org/10.1016/j.physrep.2019.03.001) (2019).
