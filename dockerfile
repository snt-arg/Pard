FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# Keys
ARG ssh_prv_key
ARG ssh_pub_key

# Work keys
RUN  apt-get -yq update && apt-get -yqq install ssh
RUN mkdir -p -m 0700 /root/.ssh && \
ssh-keyscan -H github.com >> /root/.ssh/known_hosts

RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
chmod 600 /root/.ssh/id_rsa && \
chmod 600 /root/.ssh/id_rsa.pub

# Install git
RUN apt-get update && apt-get install -y git

#  Creating working spaces directory and clone PARD inside it
RUN mkdir -p /root/workspaces
WORKDIR /root/workspaces
RUN git clone -b feat/scene_graphs git@github.com:snt-arg/Pard.git

# Install Conda
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV CONDA_DIR /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O /opt/conda/miniconda.sh \ && bash /opt/conda/miniconda.sh -b -p /opt/miniconda 
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init

#  Clone moses inside it CAN I SKIP THIS?
#RUN git clonehttps://github.com/molecularsets/moses.git

# Install graph reasoning and deps
WORKDIR /root/workspaces
RUN git clone -b feat/pard git@github.com:snt-arg/reasoning_ws.git
WORKDIR /root/workspaces/reasoning_ws
RUN pip install vcstool
RUN ./setup.sh
# RUN pip install .

# # Install graph wrapper and deps
# WORKDIR /root/workspaces
# RUN git clone -b feat/pard git@github.com:snt-arg/situational_graphs_wrapper.git
# RUN mv /root/workspaces/situational_graphs_wrapper /root/workspaces/graph_wrapper
# WORKDIR /root/workspaces/graph_wrapper
# RUN pip install .

# # Install graph matching and deps
# WORKDIR /root/workspaces
# RUN git clone -b feat/params_grid_search git@github.com:snt-arg/graph_matching.git
# WORKDIR /root/workspaces/graph_matching
# RUN pip install .
# RUN pip install transforms3d

# # Install graph reasoning and deps
# WORKDIR /root/workspaces
# RUN git clone -b train/bigger_rooms git@github.com:snt-arg/situational_graphs_reasoning.git
# RUN mv /root/workspaces/situational_graphs_reasoning /root/workspaces/graph_reasoning
# WORKDIR /root/workspaces/graph_reasoning
# RUN pip install .
# # RUN pip install transforms3d

# Conda environment and dependencies
RUN /root/workspaces/Pard/setup.sh

# Leave the workindir on Pard
WORKDIR /root/workspaces/Pard

# Removing SSH Host authorization (GitHub)
RUN rm -rf /root/.ssh/

# Shell commands in Remote machine
# python main.py device 0 dataset scene_graphs task local_denoising diffusion.num_steps 20

#### Shell commands in Local machine
## docker build --build-arg ssh_prv_key="$(cat ~/.ssh/id_ed25519)" --build-arg ssh_pub_key="$(cat ~/.ssh/id_ed25519.pub)"  -t pard:original .
### docker save pard:original -o pard.tar
### singularity -d build pard.sif docker-archive://pard.tar
## singularity -d build pard.sif docker-daemon://pard:original
## rsync --rsh='ssh -p 8022' -avzu pard.sif  jmillan@access-iris.uni.lu:workspace
## ssh iris-cluster

#### Shell command in iris machine
## si
## lsi
## rsi
## sjob