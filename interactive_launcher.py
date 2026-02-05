#!/usr/bin/env python3
"""
Interactive BindCraft Pipeline Launcher
Select parameters and launch design via menu
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_header():
    print("\n" + "="*60)
    print("🧬 BindCraft Interactive Pipeline Launcher")
    print("="*60 + "\n")

def list_files(directory, pattern="*.json"):
    """List available config files"""
    path = Path(directory)
    if not path.exists():
        return []
    return sorted([f.name for f in path.glob(pattern)])

def display_menu(title, options):
    """Display menu and get user choice"""
    print(f"\n📋 {title}:")
    print("-" * 50)
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    
    while True:
        try:
            choice = int(input("\nSelect option (number): ").strip())
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except ValueError:
            pass
        print("Invalid choice. Try again.")

def read_config(filepath):
    """Read JSON config file"""
    with open(filepath) as f:
        return json.load(f)

def show_config_details(config):
    """Display config details"""
    print(f"\n📊 Configuration Details:")
    print(f"  Design Name: {config.get('design_name', 'N/A')}")
    print(f"  PDB: {config.get('starting_pdb', 'N/A')}")
    print(f"  Chain: {config.get('chain', 'N/A')}")
    print(f"  Hotspots: {config.get('target_hotspot_residues', 'N/A')}")
    print(f"  Peptide lengths: {config.get('lengths', [])}")
    print(f"  Final designs: {config.get('number_of_final_designs', 'N/A')}")
    print(f"  Trajectories: {config.get('number_of_trajectories', 'N/A')}")

def launch_pipeline(target, algorithm, filters):
    """Launch pipeline via Docker"""
    cmd = [
        "bash",
        "-c",
        f"""
export CUDA_VISIBLE_DEVICES=1
cd /home/vincent/code/repo/biotech/BindCraft
./launch_pipeline.sh {target} {algorithm} {filters}
"""
    ]
    
    print(f"\n🚀 Launching pipeline on GPU RTX 4090 (CUDA:1)...")
    subprocess.run(cmd, cwd="/home/vincent/code/repo/biotech/BindCraft")

def main():
    os.chdir("/home/vincent/code/repo/biotech/BindCraft")
    
    print_header()
    
    # 1. Select TARGET
    targets = list_files("settings_target")
    if not targets:
        print("❌ No target configs found in settings_target/")
        sys.exit(1)
    
    print("🎯 Select Target Protein:")
    target_config = display_menu("Available Targets", targets)
    target_path = f"settings_target/{target_config}"
    
    target_data = read_config(target_path)
    show_config_details(target_data)
    
    # Confirm target
    confirm = input("\n✓ Continue with this target? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # 2. Select ALGORITHM
    algorithms = list_files("settings_advanced")
    if not algorithms:
        print("❌ No algorithm configs found in settings_advanced/")
        sys.exit(1)
    
    print("\n🧠 Select Design Algorithm:")
    print("  (Recommended for peptides: peptide_3stage_multimer.json)")
    algorithm_config = display_menu("Available Algorithms", algorithms)
    algorithm_path = f"settings_advanced/{algorithm_config}"
    
    # 3. Select FILTERS
    filters_list = list_files("settings_filters")
    if not filters_list:
        print("❌ No filters configs found in settings_filters/")
        sys.exit(1)
    
    print("\n🔍 Select Quality Filters:")
    filters_config = display_menu("Available Filters", filters_list)
    filters_path = f"settings_filters/{filters_config}"
    
    # Summary
    print("\n" + "="*60)
    print("📋 LAUNCH SUMMARY")
    print("="*60)
    print(f"Target:    {target_config}")
    print(f"Algorithm: {algorithm_config}")
    print(f"Filters:   {filters_config}")
    print("="*60)
    
    # Final confirmation
    confirm = input("\n🚀 Launch pipeline? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Launch
    launch_pipeline(target_path, algorithm_path, filters_path)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
