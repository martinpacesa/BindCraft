#!/usr/bin/env python3
"""
Wrapper to call PyRosetta from conda env when available
"""
import subprocess
import sys
import os

def has_pyrosetta_env():
    """Check if PyRosetta conda env exists"""
    return os.path.exists('/opt/miniforge/envs/pyrosetta/bin/python')

def call_pyrosetta_relax(pdb_file, relaxed_pdb_path):
    """Call PyRosetta relax from conda env"""
    if not has_pyrosetta_env():
        return False
    
    script = f"""
import sys
sys.path.insert(0, '/workspace/BindCraft/functions')
from pyrosetta_utils_native import pr_relax_native
pr_relax_native('{pdb_file}', '{relaxed_pdb_path}')
"""
    
    try:
        subprocess.run(
            ['/opt/miniforge/envs/pyrosetta/bin/python', '-c', script],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: pyrosetta_wrapper.py <input.pdb> <output.pdb>")
        sys.exit(1)
    
    success = call_pyrosetta_relax(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
