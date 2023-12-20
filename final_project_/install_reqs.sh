#!/bin/bash

# works w/ version >=525.60.13 cuda drivers, can check w/ cat /proc/driver/nvidia/version
conda install python==3.11.0
conda install pip
pip install torch torchvision torchaudio
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install matplotlib
pip install optax flax
pip install -Uq tfp-nightly[jax] > /dev/null
pip install numpy
pip install ml_collections
pip install absl-py
pip install tqdm
pip install h5py

# ptxas bug?
conda install -c nvidia cuda-nvcc