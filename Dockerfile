FROM mambaorg/micromamba:2.3-cuda12.8.1-ubuntu24.04

# Add OCI compliant labels
LABEL org.opencontainers.image.title="BindCraft"
LABEL org.opencontainers.image.description="Simple binder design pipeline using AlphaFold2 backpropagation, MPNN, and PyRosetta"
LABEL org.opencontainers.image.url="https://github.com/martinpacesa/BindCraft"
LABEL org.opencontainers.image.source="https://github.com/martinpacesa/BindCraft"
LABEL org.opencontainers.image.vendor="BindCraft"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="BindCraft Team"

USER root

ENV ENV_NAME=BindCraft
ENV PYTHON_VERSION=3.10
ENV AF2_PARAMS="alphafold_params_2022-12-06"

# Install system packages, create environment, and install all conda packages in one layer
COPY environment.yml .
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget tar libgfortran5 fswatch curl unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    micromamba config set use_lockfiles False && \
    micromamba create --name ${ENV_NAME} python=${PYTHON_VERSION} -y --file environment.yml && \
    micromamba clean --all --yes

# From now on, activate the mamba environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Set Installation directory for BindCraft
WORKDIR /opt/${ENV_NAME}

# Install ColabDesign, download AF2 weights, and install AWS CLI in one layer
ENV params_dir="params"
ENV params_file="${params_dir}/${AF2_PARAMS}.tar"
RUN pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps && \
    mkdir -p "${params_dir}" && \
    wget -q -O "${params_file}" "https://storage.googleapis.com/alphafold/${AF2_PARAMS}.tar" && \
    tar -xf "${params_file}" -C "${params_dir}" && \
    rm "${params_file}"


# Add AWS CLI
#RUN curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
#    unzip -q awscliv2.zip && \
#    ./aws/install && \
#    rm -rf awscliv2.zip aws

# Add gcloud CLI
#RUN curl -s "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz" -o "google-cloud-cli-linux-x86_64.tar.gz" && \
#    tar -xf google-cloud-cli-linux-x86_64.tar.gz && \
#    ./google-cloud-sdk/install.sh --quiet && \
#    echo 'export PATH=${PATH}:/root/google-cloud-sdk/bin' >> .bashrc

# Add Azure CLI
#RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash


# Copy BindCraft directories and set permissions
COPY . .
RUN chmod +x "functions/dssp" "functions/DAlphaBall.gcc"

# If you want to customize the ENTRYPOINT and have a conda
# environment activated, then do this:
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["/bin/bash", "-c", "/usr/local/bin/_entrypoint.sh", "python -u bindcraft.py"]