#!/bin/bash

conda create -n plantcapnet python==3.10 r
conda activate plantcapnet
conda install -c conda-forge r-vegan
pip install -r requirements.txt
pip install -r plantcapnet_compute/requirements.txt