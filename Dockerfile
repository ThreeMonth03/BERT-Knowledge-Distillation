FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

ARG USER=docker
ARG GROUP=docker
ARG PASSWORD=docker
ARG HOME=/home/${USER}

ARG UID=1000
ARG GID=1000

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="charles"

RUN groupadd -g ${GID} ${GROUP} && useradd -m ${USER} --uid=${UID} --gid=${GID} && echo "${USER}:${PASSWORD}" | chpasswd

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install -y git locales tzdata sudo && adduser ${USER} sudo
RUN apt-get clean

RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN python3 -m pip install bpython pyyaml torchinfo
RUN locale-gen en_US.UTF-8

USER ${UID}:${GID}
WORKDIR ${HOME}

RUN echo "export LC_ALL=en_US.UTF-8" >> .bashrc
RUN echo "export TERM=xterm-256color" >> .bashrc
