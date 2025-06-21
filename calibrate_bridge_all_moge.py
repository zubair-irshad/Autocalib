import os
import glob
import argparse
import subprocess
from tqdm import tqdm
import pickle
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def set_nested_dict(d, keys, value):
    """Set a value in a nested dictionary, creating intermediate dictionaries as needed."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def get_nested_dict(d, keys):
    """Get a value from a nested dictionary, return None if any key is missing."""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return None
    return d

# python calibrate_bridge_all_moge.py

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Calibrate all bridge data folders with moge processing.")
parser.add_argument('--raw_path', type=str, default="/data/group_data/katefgroup/datasets/bridge_chenyu/raw")
parser.add_argument('--dict_path', type=str, default="/data/group_data/katefgroup/datasets/bridge_chenyu/raw/calib_dict.pkl")
parser.add_argument('--traj_paths', type=str, default="/data/group_data/katefgroup/datasets/bridge_chenyu/raw/traj_paths.txt",
                    help="Path to file containing trajectory paths, if exists")
parser.add_argument('--verbose', action='store_true', help="Print subprocess output")
parser.add_argument('--moge_env', type=str, default="mega_sam", help="Conda environment for moge_static_bridge.py")
parser.add_argument('--moge_script_path', type=str, default="/data/user_data/wenhsuac/chenyuzhang/moge_video/moge_static_bridge.py")
parser.add_argument('--calib_script_path', type=str, default="/data/user_data/wenhsuac/chenyuzhang/backup/drrobot/calibrate_bridge_single.py")

args = parser.parse_args()

# Load existing dictionary or initialize a new one
if os.path.exists(args.dict_path):
    with open(args.dict_path, 'rb') as f:
        calib_dict = pickle.load(f)
    print(f"Loaded existing dictionary from {args.dict_path}")
else:
    calib_dict = {}
    print(f"Initialized new dictionary at {args.dict_path}")

# # Find all data folders
# wildcards = ['*'] * 5
# pattern = os.path.join(args.raw_path, *wildcards, 'raw', 'traj_group*', 'traj0*')
# traj_paths = glob.glob(pattern)
# print('Found raw traj paths:', len(traj_paths))
# traj_paths = sorted(traj_paths, key=lambda p: [natural_sort_key(part) for part in p.split(os.sep)])
if os.path.exists(args.traj_paths):
    with open(args.traj_paths, 'r') as f:
        traj_paths = [line.strip() for line in f.readlines()]
    print('Loaded traj paths from file:', len(traj_paths))
else:
    # Your original code to collect and sort traj_paths
    wildcards = ['*'] * 5
    pattern = os.path.join(args.raw_path, *wildcards, 'raw', 'traj_group*', 'traj0*')
    traj_paths = glob.glob(pattern)
    print('Found raw traj paths:', len(traj_paths))
    traj_paths = sorted(traj_paths, key=lambda p: [natural_sort_key(part) for part in p.split(os.sep)])
    
    # Save the paths to a file for future use
    with open(args.traj_paths, 'w') as f:
        for path in traj_paths:
            f.write(path + '\n')
print(f"{len(traj_paths)} trajectories found")
print(traj_paths[:100])  # Print first two paths for verification
# Extract moge project directory from script path
moge_project_dir = os.path.dirname(args.moge_script_path)
calib_project_dir = os.path.dirname(args.calib_script_path)

# Process each data folder with progress bar
for traj_path in tqdm(traj_paths[1400:], desc="Processing trajectories"):
    # Compute dictionary keys from relative path
    rel_path = os.path.relpath(traj_path, args.raw_path)
    keys = rel_path.split(os.sep)
    
    # Skip if already processed
    if get_nested_dict(calib_dict, keys) is not None:
        if args.verbose:
            print(f"Skipping {traj_path}, already processed")
        continue
    
    if args.verbose:
        print(f"Processing {traj_path}")
    
    # Run moge_static_bridge.py in its environment and project directory
    moge_cmd = [
        'conda', 'run', '-n', args.moge_env, 'python', args.moge_script_path,
        '--scene_path', traj_path
    ]
    try:
        if args.verbose:
            subprocess.run(moge_cmd, check=True, cwd=moge_project_dir)
        else:
            subprocess.run(moge_cmd, check=True, cwd=moge_project_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error running moge for {traj_path}: {e}")
        continue  # Skip to next trajectory if moge fails
