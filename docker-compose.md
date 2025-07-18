# BindCraft Docker Compose Setup

This docker-compose configuration allows you to run BindCraft in a containerized environment with GPU support.

## Prerequisites

- Docker Engine with Compose V2
- NVIDIA Container Toolkit installed
- NVIDIA GPU with compatible drivers

## Usage

1. Make sure your target settings are in the `settings_target` directory
2. Create the necessary directories:
   ```bash
   mkdir -p results
   ```
3. Place your target PDB file in the `example` directory
4. Create your target JSON file in `settings_target` directory
5. Run the container:
   ```bash
   docker compose up
   ```

## Configuration

The docker-compose.yml file mounts three directories:
- `./results`: Where design results will be saved
- `./settings_target`: Where your target JSON files are located
- `./example`: Where your target PDB files are located

You can modify the command parameters in the docker-compose.yml file to use different settings files.

## Example Target JSON

Create a file like `settings_target/PDL1.json` with content:

```json
{
    "design_path": "./results/PDL1/",
    "binder_name": "PDL1",
    "starting_pdb": "./example/PDL1.pdb",
    "chains": "A",
    "target_hotspot_residues": "56",
    "lengths": [65, 150],
    "number_of_final_designs": 100
}
```

## Troubleshooting

If you encounter GPU-related issues, ensure the NVIDIA Container Toolkit is properly installed:

```bash
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```