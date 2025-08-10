import h5py
import numpy as np

def save_last_frames(full_data_path, poses_data_path):
    """
    Save the last frame of each trajectory to a poses file.
    
    Args:
        full_data_path: str - Path to full trajectory data file
        poses_data_path: str - Path to save poses file
    """
    last_frames_object = []
    last_frames_robot = []
    grid_points = []
    finger_poses = []
    
    # Read all episodes from full data file
    with h5py.File(full_data_path, 'r') as f:
        episode_keys = [key for key in f.keys() if key.startswith('episode_')]
        
        for episode_key in episode_keys:
            episode_group = f[episode_key]

            finger_pos = episode_group.attrs['finger_pos']
            if finger_pos > 0.8: # grasping failed, skip
                continue 

            object_data = episode_group['object'][:]
            robot_data = episode_group['robot'][:]
            grid_point = episode_group['grid_point'][:]

            # Get the last frame
            last_frame_object = object_data[-1]  # [n_obj, 3]
            last_frame_robot = robot_data[-1]    # [n_robot, 3]
            
            last_frames_object.append(last_frame_object)
            last_frames_robot.append(last_frame_robot)
            grid_points.append(grid_point)
            finger_poses.append(finger_pos)
    
    last_frames_object = np.stack(last_frames_object, axis=0)
    last_frames_robot = np.stack(last_frames_robot, axis=0)
    grid_points = np.stack(grid_points, axis=0)
    finger_poses = np.stack(finger_poses, axis=0)
    print(grid_points.shape)

    # Save poses file with last frames
    with h5py.File(poses_data_path, 'w') as f:
        assert last_frames_object.shape[0] == last_frames_robot.shape[0] == finger_poses.shape[0] == grid_points.shape[0]
        f.attrs['n_episodes'] = last_frames_object.shape[0]
        f.create_dataset('object', data=last_frames_object)
        f.create_dataset('robot', data=last_frames_robot)
        f.create_dataset('finger', data=finger_poses)
        f.create_dataset('target', data=grid_points)
        print(f"Saved {last_frames_object.shape[0]} episodes to {poses_data_path}")

def save_episode_data(data_file_path, episode_id, object_data, robot_data, grid_point=None, finger_pos=None, gaussians_data=None):
    """
    Originally intended to be a general function to save data, but now only used by start_poses.py
    Args:
        data_file_path: str - Path of the data file to save to
        episode_id: int - Episode ID
        object_data: list of tensors of shape [n_particles, 3] - object point cloud
        robot_data: list of tensors of shape [n_particles, 3] - robot point cloud
        grid_point: torch.Tensor [3] - Grid point that the robot starts at
        finger_pos: float - Finger position
        gaussians_data: list of dictionaries - 3D gaussian data. DEPRECATED
    """
    
    # Convert to numpy arrays
    object_array = np.array([x.detach().cpu().numpy() for x in object_data])
    robot_array = np.array([x.detach().cpu().numpy() for x in robot_data])
    grid_point = grid_point.detach().cpu().numpy() if grid_point is not None else None
    
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

        if grid_point is not None:
            episode_group.create_dataset(
                'grid_point',
                data=grid_point,
            )
        
        # Gaussians data (if available)
        if gaussians_data is not None:
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
        episode_group.attrs['episode_id'] = episode_id
        episode_group.attrs['include_gaussian'] = gaussians_data is not None
        if finger_pos is not None:
            episode_group.attrs['finger_pos'] = finger_pos
    
    print(f"Saved episode {episode_id} to {data_file_path}: object {object_array.shape}, robot {robot_array.shape}")
