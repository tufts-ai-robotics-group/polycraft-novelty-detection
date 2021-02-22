# polycraft-novelty-detection
Visual novelty detector for the Polycraft domain.

## Installation
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
pipenv run pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## Testing
To run unit tests, run the following command:
```
pipenv run python -m pytest
```