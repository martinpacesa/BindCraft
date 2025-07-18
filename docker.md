# BindCraft Docker Guide

This guide will help you build and run BindCraft using Docker, even if you're new to Docker technology.

> **DISCLAIMER**: This Dockerfile is provided for personal and research use only. Before distributing any Docker image built using this Dockerfile, please verify that you have the appropriate licenses for all included software components (AlphaFold2, PyRosetta, ColabDesign, ProteinMPNN, etc.). The authors of BindCraft do not take any responsibility for the distribution of Docker images built using this Dockerfile. It is the user's responsibility to ensure compliance with all applicable licenses and terms of use.

## What is Docker?

Docker is a tool that packages software into standardized units called "containers" that include everything needed to run the software: code, runtime, system tools, and libraries. This ensures that BindCraft will run the same way regardless of your computer setup.

## Prerequisites

Before you begin, you need to install:

1. **Docker Desktop**:
   - For Windows/Mac: Download and install from [Docker's website](https://www.docker.com/products/docker-desktop)
   - For Linux: Follow the [installation instructions](https://docs.docker.com/engine/install/) for your distribution

2. **NVIDIA Container Toolkit**
   - Follow the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

3. **NVCC**
   - For Windows: Follow the [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
   - For Linux: Follow the [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

You can check that all the above-mentioned components are correctly installed by issuing the following commands: 

```bash
nvcc --version
#nvcc: NVIDIA (R) Cuda compiler driver
#Copyright (c) 2005-2023 NVIDIA Corporation
#Built on Fri_Jan__6_16:45:21_PST_2023
#Cuda compilation tools, release 12.0, V12.0.140
#Build cuda_12.0.r12.0/compiler.32267302_0
```
and

```bash
$ docker --version
#Docker version 28.0.4, build b8034c0
```

## Building the BindCraft Docker Image

### Step 1: Clone the Repository

Open a terminal (Command Prompt or PowerShell on Windows) and run:

```bash
git clone https://github.com/martinpacesa/BindCraft
cd BindCraft
```

### Step 2: Build the Docker Image

Run the following command to build the Docker image:

```bash
docker build -t bindcraft:latest .
```

Depending on the hardware characteristics of your machine, this process will take some time (10-15 minutes) as it downloads and installs all necessary components.

## Running BindCraft

If it is the first time that you've run docker containers, you might need to run the following command to configure the NVIDIA Container Toolkit: 

```bash
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### Option 1: Using Docker Run (Basic)

1. Create directories for your data:

```bash
mkdir -p results
```

2. Place your target PDB file in the `example` directory or use the `PDL1.pdb` file provided as example 

3. Create a settings file in the `settings_target` directory (e.g., `settings_target/my_target.json`). The following file was created for you:

```json
{
    "design_path": "./results/PDL1/",
    "binder_name": "PDL1",
    "starting_pdb": "./example/PDL1.pdb",
    "chains": "A",
    "target_hotspot_residues": "56",
    "lengths": [65, 150],
    "number_of_final_designs": 5
}
```

4. Run BindCraft:

```bash
docker run --gpus all \
  -v ./results:/opt/BindCraft/results \
  -v ./settings_target:/opt/BindCraft/settings_target \
  -v ./example:/opt/BindCraft/example \
  bindcraft:latest \
  python -u bindcraft.py \
  --settings './settings_target/PDL1.json' \
  --filters './settings_filters/default_filters.json' \
  --advanced './settings_advanced/default_4stage_multimer.json'
```

### Option 2: Using Docker Compose (Recommended)

1. Create the same directories and files as in Option 1

2. Run:

```bash
docker compose up
```

## Troubleshooting

### GPU Issues

If you encounter GPU-related errors:

```bash
# For Linux
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Permission Issues

If you encounter permission issues with the results directory:

```bash
# For Linux/Mac
chmod -R 777 results
```

### Docker Not Found

If you get "command not found" errors:

- Make sure Docker Desktop is running
- Try restarting your computer
- For Linux, you might need to add your user to the docker group:
  ```bash
  sudo usermod -aG docker $USER
  # Then log out and back in
  ```

## Understanding the Results

After running BindCraft, your results will be in the `results` directory. The structure will be:

```
results/
└── my_target/
    ├── Accepted/       # Successful designs
    ├── MPNN/           # MPNN-generated sequences
    ├── Rejected/       # Failed designs
    ├── Trajectory/     # Design trajectories
    ├── final_design_stats.csv  # Summary statistics
    ├── failure_csv.csv         # Summary of the failed designs
    ├── mpnn_design_stats.csv   # Summary statistics of MPNN designs
    └── trajectories_stats.csv  # Summary statistics of found trajectories
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [BindCraft Documentation](https://github.com/martinpacesa/BindCraft)
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)

## Support

If you encounter issues with BindCraft, please open an issue on the [GitHub repository](https://github.com/martinpacesa/BindCraft/issues).