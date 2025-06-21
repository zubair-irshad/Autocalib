import os
import argparse
import subprocess
import re
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Calibrate all droid scenes and save results to a dictionary.")
    parser.add_argument('--batch_path', type=str, default="/data/group_data/katefgroup/datasets/droid_chenyu/droid_extract_whole",
                        help="Path to the droid_extract_whole directory")
    parser.add_argument('--dict_path', type=str, default="/data/group_data/katefgroup/datasets/droid_chenyu/droid_extract_whole/calib_dict.pkl",
                        help="Path to the calibration dictionary pickle file")
    parser.add_argument('--verbose', action='store_true', help="Print subprocess output")
    parser.add_argument('--start_index', type=int, default=0, help="Start processing from this index")
    parser.add_argument('--conda_env', type=str, default="dr_test", help="Conda environment for the calibration script")
    parser.add_argument('--calib_script_path', type=str, default="/data/user_data/wenhsuac/chenyuzhang/backup/drrobot/calibrate_droid_single_alltracker.py",
                        help="Path to the calibration script")
    args = parser.parse_args()

    # List all scene directories under batch_path
    scene_dirs = [d for d in os.listdir(args.batch_path) 
                  if os.path.isdir(os.path.join(args.batch_path, d)) and re.match(r'scene_\d+', d)]
    
    # Sort directories numerically based on the number after "scene_"
    scene_dirs.sort(key=lambda d: int(d.split('_')[1]))
    
    # Apply start_index to skip earlier scenes
    scene_dirs = scene_dirs[args.start_index:]

    # Hardcoded model path from the bash script
    model_path = "output/franka_fr3_2f85_highres_finetune_0"

    # Calibration project directory (where the calibration script resides)
    calib_project_dir = os.path.dirname(args.calib_script_path)

    # Process each scene with a progress bar
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_path = os.path.join(args.batch_path, scene_dir)
        if args.verbose:
            print(f"Processing {scene_path}")

        # Command to run the calibration script
        calib_cmd = [
            'conda', 'run', '-n', args.conda_env, 'python', args.calib_script_path,
            '--model_path', model_path,
            '--scene_path', scene_path,
            '--dict_path', args.dict_path
        ]

        try:
            if args.verbose:
                subprocess.run(calib_cmd, check=True, cwd=calib_project_dir)
            else:
                subprocess.run(calib_cmd, check=True, cwd=calib_project_dir, 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {scene_path}: {e}")

if __name__ == "__main__":
    main()