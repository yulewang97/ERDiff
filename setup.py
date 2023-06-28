#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
        "torch",
        "torchvision",
        "h5py",
        "seaborn",
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "jupyter",
        "tqdm",
        "einops",
        "POT"
]

setup(
    name='erdiff',
    version='0.1.0',
    description="Extraction and recovery of spatio-temporal structure in latent dynamics alignment with diffusion model (ERDiff)",
    long_description=readme,
    author="Yule Wang",
    author_email='yulewang@gatech.edu',
    install_requires=requirements,
    license="MIT license",
    keywords='Computational Neuroscience, Neural Latent Dynamics Alignment, Diffusion Model, Neural decoding, Brain-computer Interfaces',
    packages=find_packages(include=['VAE_Diffusion_CoTrain.py', 'MLA.py', 'model_functions'])
)
