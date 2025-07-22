import functools
import torch
import numpy as np
import h5py
import os
from GNN.utils import fps_rad_tensor


def get_dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def singleton(cls):
    _instance = {}

    @functools.wraps(cls)
    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner

def save_episode_data(data_file_path, episode_id, object_data, robot_data, include_gaussian=False, gaussians_data=None):
    """Save episode data to a shared HDF5 file with each episode as a group"""
    
    # Convert to numpy arrays
    object_array = np.array([x.detach().cpu().numpy() for x in object_data])
    robot_array = np.array([x.detach().cpu().numpy() for x in robot_data])
    
    # Open HDF5 file in append mode
    with h5py.File(data_file_path, 'a') as f:
        # Create episode group
        episode_group = f.create_group(f'episode_{episode_id:06d}')
        
        # Create datasets with chunk size equal to the whole dataset
        episode_group.create_dataset(
            'object', 
            data=object_array, 
            compression='gzip', 
            compression_opts=9,
            shuffle=True,
            chunks=object_array.shape  # Chunk size equal to whole dataset
        )
        
        episode_group.create_dataset(
            'robot', 
            data=robot_array, 
            compression='gzip', 
            compression_opts=9,
            shuffle=True,
            chunks=robot_array.shape  # Chunk size equal to whole dataset
        )
        
        # Gaussians data (if available)
        if include_gaussian and gaussians_data:
            gaussians = episode_group.create_group('gaussians')
            
            # Extract arrays from gaussians data
            xyz_arrays = []
            rotation_arrays = []
            frame_counts = []
            for gaussians_data_item in gaussians_data:
                xyz_arrays.append(gaussians_data_item['xyz'].detach().cpu().numpy())
                rotation_arrays.append(gaussians_data_item['rotation'].detach().cpu().numpy())
                frame_counts.append(gaussians_data_item['frame_count'])
            
            xyz_data = np.array(xyz_arrays)
            rotation_data = np.array(rotation_arrays)
            frame_counts_data = np.array(frame_counts)
            
            gaussians.create_dataset(
                'xyz', 
                data=xyz_data, 
                compression='gzip', 
                compression_opts=9,
                shuffle=True,
                chunks=xyz_data.shape
            )
            gaussians.create_dataset(
                'rotation', 
                data=rotation_data, 
                compression='gzip', 
                compression_opts=9,
                shuffle=True,
                chunks=rotation_data.shape
            )
            gaussians.create_dataset(
                'frame_counts', 
                data=frame_counts_data, 
                compression='gzip', 
                compression_opts=9
            )
        
        # Store metadata as attributes
        episode_group.attrs['n_frames'] = len(object_data)
        episode_group.attrs['n_obj_particles'] = object_array.shape[1]
        episode_group.attrs['n_bot_particles'] = robot_array.shape[1]
        episode_group.attrs['object_type'] = 'rope'  # As specified in requirements
        episode_group.attrs['motion_type'] = 'single_push'  # As specified in requirements
        episode_group.attrs['episode_id'] = episode_id
        episode_group.attrs['include_gaussian'] = include_gaussian
    
    print(f"Saved episode {episode_id} to {data_file_path}: object {object_array.shape}, robot {robot_array.shape}")


