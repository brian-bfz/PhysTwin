import h5py
import argparse
from shared.utils import parse_episodes
import random
import os
from datetime import datetime

def merge_datasets(input_files, output_file):
    """
    Merge multiple HDF5 data files into a single file.
    
    Args:
        input_files: List of input HDF5 file paths
        output_file: Output HDF5 file path
    """
    print(f"Merging {len(input_files)} data files into {output_file}")
    
    # Collect all episodes from all files
    all_episodes = []
    
    for file_idx, data_file in enumerate(input_files):
        if not os.path.exists(data_file):
            print(f"Warning: Data file {data_file} does not exist, skipping...")
            continue
            
        with h5py.File(data_file, 'r') as f:
            # Get all episode keys from this file
            for key in f.keys():
                if key.startswith('episode_'):
                    episode_id = int(key.split('_')[1])
                    all_episodes.append((episode_id, key, file_idx))
    
    print(f"Found {len(all_episodes)} total episodes across all files")
    
    # Shuffle the episode indices for random ordering
    indices = list(range(len(all_episodes)))
    random.shuffle(indices)
    
    # Merge all episodes into the output file
    with h5py.File(output_file, 'w') as fout:
        # Add global metadata
        fout.attrs['created'] = datetime.now().isoformat()
        fout.attrs['description'] = 'PhysTwin merged episode data collection'
        fout.attrs['source_files'] = len(input_files)
        
        for count, (episode_id, episode_name, file_idx) in enumerate(all_episodes):
            new_episode_name = f'episode_{indices[count]:06d}'
            
            # Copy episode from source file
            with h5py.File(input_files[file_idx], 'r') as fsrc:
                fsrc.copy(fsrc[episode_name], fout, name=new_episode_name)
            
            if count % 100 == 0:  # Progress update every 10 episodes
                print(f"Merged {count}/{len(all_episodes)} episodes...")
    
    print(f"Successfully merged {len(all_episodes)} episodes into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all episodes from multiple H5 files into a new H5 file.")
    parser.add_argument('--data_dir', type=str, default='PhysTwin/generated_data')
    parser.add_argument('--input_files', nargs='+', type=str, required=True, 
                       help='Names of input data files (without .h5 extension)')
    parser.add_argument('--output_file', type=str, required=True, help='Name of the output data file')

    args = parser.parse_args()

    assert len(args.input_files) > 1, "Must provide at least two input files"

    # Construct full file paths
    input_files = [args.data_dir + '/' + fname + '.h5' for fname in args.input_files]
    output_file = args.data_dir + '/' + args.output_file + '.h5'
    
    # Check if input files exist
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: The following input files do not exist:")
        for f in missing_files:
            print(f"  {f}")
        exit(1)
    
    merge_datasets(input_files, output_file)