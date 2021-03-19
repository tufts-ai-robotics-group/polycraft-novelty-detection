from setuptools import setup, find_packages


setup(
    name="polycraft_nov_det",
    version="0.0.1",
    description="Polycraft novelty detection",
    url="https://github.com/tufts-ai-robotics-group/polycraft-novelty-detection",
    author="Patrick Feeney, Sarah Schneider",
    packages=find_packages(),
    python_requires=">=3.8, <4",
    install_requires=[
        "matplotlib",
        "numpy",
        "tensorboard",
        "tensorflow",
        # dev packages, not installing correctly when in extras_require
        "autopep8",
        "flake8",
        "pep8-naming",
        "pytest",
        "setuptools",
    ],
)
