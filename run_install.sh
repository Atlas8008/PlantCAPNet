#!/bin/bash

# conda create -n plantcapnet python==3.10 r
# conda install --file conda_config.txt
# conda update setuptools
# python -m ensurepip --default-pip
# pip install -r plantcapnet_compute/requirements.txt

conda create -n plantcapnet python==3.10 r
conda activate plantcapnet
conda install -c conda-forge r-vegan
pip install -r requirements.txt
pip install -r plantcapnet_compute/requirements.txt