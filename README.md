# polycraft-novelty-detection
Visual novelty detector for the Polycraft domain.

## Installation

This repository utilizes Git submodules so utilize ```git clone --recurse-submodules``` instead of ```git clone``` when first downloading the repository. If this command was forgotten when cloning, run the following from inside the repository:
```
git submodule update --init --recursive
```

If you do not have Pipenv installed, run the following:
```
pip install pipenv
```
The Pipenv dependencies can be installed within a Pipenv with the following commands:
```
pipenv install
```
Pytorch requires different versions depending on the machine it is running on. Therefore it is not included in the Pipenv by default. To install Pytorch, generate the appropriate Pip command on the [Pytorch website](https://pytorch.org/get-started/locally/) then run it within the Pipenv by prepending ```pipenv run``` to it. For example:
```
pipenv run pip3 install torch torchvision torchaudio
```

#### Conda Installation

The following instructions are **not recommended** unless you are unable to install a Python version compatible with the Pipenv.

For this installation Pytorch will be installed in the Conda environment using the appropriate command according to the [Pytorch website](https://pytorch.org/get-started/locally/). For example:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Then the Pipenv must be configured for the Conda environment before install:
```
pipenv --python=$(conda run which python) --site-packages
pipenv install
```

## Testing
To run unit tests, run the following command:
```
pipenv run python -m pytest
```

## Directory Structure

#### polycraft-novelty-detection (root)

The root folder should only contain the project README and configuration files.

#### models

Saved copies of Pytorch models whose Tensorboard data is stored in the runs folder.

#### polycraft_nov_det

Python module for Polycraft novelty detection. All Python code for the repository should be within this folder and be accessible through the Python module interface.

#### runs

Tensorboard data for the Pytorch models saved in the models folder.

#### submodules

Git submodules used as dependencies for the project.

#### tests

Automated Pytest tests for polycraft_nov_det, divided by test subject.

## Model Training

Model training can be done by executing the module. To view available configurations run:

```
pipenv run python -m polycraft_nov_det -h
```
