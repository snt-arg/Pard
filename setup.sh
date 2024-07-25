#!/bin/bash
# create conda env
ENV=pard
CUDA=11.8
cuda=cu118

# ### python 3.8 is needed for molsets if you want to install molsets with pip
conda create -n $ENV python=3.10 -y
eval "$(conda shell.bash hook)" 
conda activate $ENV

# install pytorch 2.0+ (tested with 2.2)
# conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA -c pytorch -c nvidia -y
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y 

# install rdkit
conda install -c conda-forge rdkit -y

# install graph-tool
conda install -c conda-forge graph-tool=2.45 -y


# install ipython and notebook
conda install nb_conda -c conda-forge  -y

# install packages due to error when creating the singularity
conda install -c conda-forge pluggy
conda install -c conda-forge certifi -y

# install pytorch-geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+$cuda.html

# install lightning
pip install lightning

# install other packages
pip install yacs tensorboard
pip install pyemd pygsp
pip install einops
pip install wandb
pip install pandas==1.5.3

# reasoning deps
pip install colorama
pip install seaborn

# install molsets
## if use python 3.8, you can try pip 
# pip install molsets
# ## if use other versions, or you failed installation 
# - please install manually, https://github.com/molecularsets/moses#manually