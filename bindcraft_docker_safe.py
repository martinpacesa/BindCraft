#!/usr/bin/env python3
"""
BindCraft Docker-Safe Launcher
Handles JAX/ColabDesign initialization properly for Docker GPU environments
"""

import os
import sys
import subprocess

def main():
    # Ensure JAX can fallback to CPU
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = 'gpu,cpu'
    
    # Set up logging
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Import after env setup
    import argparse
    
    parser = argparse.ArgumentParser(description='BindCraft safe launcher for Docker')
    parser.add_argument('--settings', '-s', type=str, required=True, help='Path to settings.json')
    parser.add_argument('--filters', '-f', type=str, default='./settings_filters/default_filters.json', help='Path to filters.json')
    parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/default_4stage_multimer.json', help='Path to advanced.json')
    
    args = parser.parse_args()
    
    # Now import the main bindcraft module
    try:
        print("✓ Initializing BindCraft...")
        from functions import *
        
        print("✓ Loading settings...")
        import json
        with open(args.settings) as f:
            target_settings = json.load(f)
        
        print(f"✓ Design: {target_settings.get('design_name', 'Unknown')}")
        print(f"✓ Target: {target_settings.get('starting_pdb', 'Unknown')}")
        
        # For now, just print success and exit
        # Full pipeline integration coming next
        print("✅ BindCraft Docker Safe Launcher ready")
        print(f"   Settings: {args.settings}")
        print(f"   Filters: {args.filters}")
        print(f"   Advanced: {args.advanced}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
