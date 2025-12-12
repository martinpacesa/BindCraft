ARG APP=bindcraft
ARG BUILDPLATFORM=linux/amd64
ARG TARGETARCH=amd64

# cbe cluster 2024-10
ARG CUDA_VERSION="12.2"

# 3.10 because bindcraft uses 3.10
ARG MINICONDA_VERSION="Miniconda3-py310_24.7.1-0-Linux-x86_64"

ARG USER_FOLDER="/pub"
ARG CONDA_INSTALL_PATH="/pub/conda"
ARG CONDA_ENV_NAME="bindcraft_env"

ARG BINDCRAFT_GIT_REPO="https://github.com/xeniorn/BindCraft"

####################################################################################
FROM --platform=$BUILDPLATFORM debian:stable-slim AS builder

ARG TARGETARCH
ARG APP

RUN dpkg --add-architecture $TARGETARCH \
    && apt-get update \
    && apt-get install -y \
      less wget git \
    && true
      # && rm -rf /var/lib/apt/lists/*

#="/pub/conda"
ARG CONDA_INSTALL_PATH
# ="Miniconda3-py311_24.1.2-0-Linux-x86_64"
ARG MINICONDA_VERSION

RUN wget --quiet \
    https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}.sh \
     \
    && bash ${MINICONDA_VERSION}.sh -bfp ${CONDA_INSTALL_PATH} \
    && rm -f ${MINICONDA_VERSION}.sh 

ENV PATH="${CONDA_INSTALL_PATH}/bin:${PATH}"

ARG BINDCRAFT_GIT_REPO
ARG CUDA_VERSION

WORKDIR /pub
RUN git clone $BINDCRAFT_GIT_REPO
WORKDIR /pub/BindCraft
RUN bash install_bindcraft.sh --pkg_manager 'conda' --cuda $CUDA_VERSION

RUN apt-get install -y libgfortran5