from ..qqtt import InvPhyTrainerWarp
from ..config_manager import PhysTwinConfig
import argparse
import torch
import numpy as np
import os
import h5py
import pickle
from ..paths import GENERATED_DATA_DIR

def create_push_action(grid_point, wait_frames):
    """
    Create an action function that positions the robot at a specific grid point and stays stationary.
    
    Args:
        grid_point: torch.Tensor [3] - target position for robot center
        wait_frames: int - number of frames to wait stationary
        
    Returns:
        function: Action function compatible with InvPhyTrainerWarp.generate_data
    """
    def push_action(init_vertices, robot_controller, n_ctrl_parts):
        # Robot starts closed (finger = 0.0)
        initial_finger = 0.0
        finger_changes = torch.zeros(wait_frames, dtype=torch.float32, device=robot_controller.device)
        
        # Calculate initial translation to move robot to grid point
        current_robot_center = robot_controller.get_current_center()
        initial_translation = grid_point - current_robot_center
        
        # No movement during wait frames
        target_changes = torch.zeros((wait_frames, n_ctrl_parts, 3), dtype=torch.float32, device=robot_controller.device)
        
        return initial_translation, target_changes, initial_finger, finger_changes
    
    return push_action

def create_lift_action(grid_point, wait_frames):
    """
    Create an action function that positions the robot above a grid point, lowers to grasp, and stays stationary.
    
    Args:
        grid_point: torch.Tensor [3] - target position for robot center
        wait_frames: int - number of frames to wait stationary after grasping
        
    Returns:
        function: Action function compatible with InvPhyTrainerWarp.generate_data
    """
    grid_point[2] += -0.005 # offset so robot doens't start too low. 
    def lift_action(init_vertices, robot_controller, n_ctrl_parts):
        # Total frames: approach + grasp + wait
        approach_frames = 10
        offset = -0.05 # z axis is reversed for phystwin
        grasp_frames = 20
        total_frames = approach_frames + grasp_frames + wait_frames
        
        # Robot starts open (finger = 1.0)
        initial_finger = 1.0
        finger_changes = torch.zeros(total_frames, dtype=torch.float32, device=robot_controller.device)
        
        # Close fingers during grasp phase
        finger_changes[approach_frames:approach_frames + grasp_frames] = -0.05
        
        # Calculate initial translation to position robot above the point
        current_robot_center = robot_controller.get_current_center()
        # Position 0.1 units above the target point
        start_position = grid_point.clone()
        start_position[2] += offset
        initial_translation = start_position - current_robot_center
        
        # Movement sequence: approach, grasp, wait
        target_changes = torch.zeros((total_frames, n_ctrl_parts, 3), dtype=torch.float32, device=robot_controller.device)
        
        # Approach: move down to the target point
        approach_speed = offset / approach_frames
        target_changes[:approach_frames, 0, 2] = -approach_speed
        
        # Grasp: wait for gripper to close
        # Wait: wait for object to stabilize
        
        return initial_translation, target_changes, initial_finger, finger_changes
    
    return lift_action

def init_phystwin(case_name):
    """
    Initialize PhysTwin trainer and simulator for a given case.
    
    Args:
        case_name: str - Name of the PhysTwin case/experiment
        
    Returns:
        tuple: (trainer, best_model_path, config)
    """
    # Setup configuration
    config = PhysTwinConfig(case_name=case_name, inference=True)
    
    # Create trainer
    trainer = InvPhyTrainerWarp(
        data_path=config.get_data_path(),
        base_dir=str(config.case_paths['base_dir']),
        pure_inference_mode=True,
        static_meshes=[],
        robot_controller=config.get_robot_controller("default", n_ctrl_parts=1, device='cuda'),
        include_gaussian=False,
        device='cuda',
    )

    # Get initial vertices
    with open(config.get_data_path(), 'rb') as f:
        data = pickle.load(f)
    # print(data.keys())
    object_vertices = torch.tensor(data['object_points'][0], dtype=torch.float32, device='cuda')
    # print(object_vertices.shape)

    return trainer, config, object_vertices

def save_last_frames(full_data_path, poses_data_path, target_points, finger_pos):
    """
    Save the last frame of each trajectory to a poses file.
    
    Args:
        full_data_path: str - Path to full trajectory data file
        poses_data_path: str - Path to save poses file
        target_points: list - List of starting points for the robot
    """
    last_frames_object = []
    last_frames_robot = []
    
    # Read all episodes from full data file
    with h5py.File(full_data_path, 'r') as f:
        episode_keys = [key for key in f.keys() if key.startswith('episode_')]
        
        for episode_key in episode_keys:
            episode_group = f[episode_key]
            object_data = episode_group['object'][:]
            robot_data = episode_group['robot'][:]
            
            # Get the last frame
            last_frame_object = object_data[-1]  # [n_obj, 3]
            last_frame_robot = robot_data[-1]    # [n_robot, 3]
            
            last_frames_object.append(last_frame_object)
            last_frames_robot.append(last_frame_robot)
    
    last_frames_object = np.stack(last_frames_object, axis=0)
    last_frames_robot = np.stack(last_frames_robot, axis=0)
    target_points = torch.stack(target_points, dim=0).cpu().numpy()
    finger_pos = np.stack(finger_pos, axis=0)
    # print("Shape of object, robot, and target:", last_frames_object.shape, last_frames_robot.shape, target_points.shape)

    # Save poses file with last frames
    with h5py.File(poses_data_path, 'w') as f:
        f.create_dataset('object', data=last_frames_object)
        f.create_dataset('robot', data=last_frames_robot)
        f.create_dataset('target', data=target_points)
        f.create_dataset('finger', data=finger_pos)

def generate_push_poses(config):
    """
    Generate a grid that spans the entire object + margin with cell_size x cell_size cells
    For each grid point, determine if it's at most max_dist away from an object point, and at least min_dist away from every object point
    If it is, generate a trajectory with InvPhyTrainerWarp.generate_data where the robot starts closed, centered at the point, and stays stationary for {wait} frames
    Tell generate_data to save to PhysTwin/generated_data/{case_name}/full_push_poses.h5
    Create a file PhysTwin/generated_data/{case_name}/push_poses.h5 that contains the last frame of each trajectory
    """
    case_name = config["case_name"]
    margin = config["margin"]
    cell_size = config["cell_size"]
    max_dist = config["max_dist"]
    min_dist = config["min_dist"]
    wait = config["wait"]

    print(f"Generating push poses for case: {case_name}")
    
    # Initialize PhysTwin
    trainer, config, object_vertices = init_phystwin(case_name)
    
    # Generate grid points
    print("Generating grid points...")
    # Calculate bounding box of object with margin
    min_coords = torch.min(object_vertices, dim=0)[0] - margin
    max_coords = torch.max(object_vertices, dim=0)[0] + margin
    
    # Generate grid
    x_range = torch.arange(min_coords[0], max_coords[0] + cell_size, cell_size)
    y_range = torch.arange(min_coords[1], max_coords[1] + cell_size, cell_size)
    z = (min_coords[2] + max_coords[2]) / 2
    
    valid_points = []
    finger_pos = []
    
    for x in x_range:
        for y in y_range:
            grid_point = torch.tensor([x, y, z], dtype=torch.float32, device=object_vertices.device)
                
            # Check distance constraints
            distances = torch.norm(object_vertices - grid_point, dim=1)
            min_distance = torch.min(distances).item()
                
            # Point is valid if it's close enough to at least one object point
            # and far enough from all object points
            if min_distance <= max_dist and min_distance >= min_dist:
                valid_points.append(grid_point)

    print(f"Found {len(valid_points)} valid grid points")
    
    # Setup output directories
    output_dir = GENERATED_DATA_DIR / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_data_path = str(output_dir / "full_push_poses.h5")
    poses_data_path = str(output_dir / "push_poses.h5")
    
    # Initialize data files
    if os.path.exists(full_data_path):
        os.remove(full_data_path)
    if os.path.exists(poses_data_path):
        os.remove(poses_data_path)
    
    # Generate trajectories for each grid point
    for i, grid_point in enumerate(valid_points):
        print(f"Processing grid point {i+1}/{len(valid_points)}: {grid_point.cpu().numpy()}")
                
        # Generate trajectory
        trainer.generate_data(
            config.get_best_model_path(),
            create_push_action(grid_point, wait),
            config.get_gaussian_path(),
            n_ctrl_parts=1,
            data_file_path=full_data_path,
            episode_id=i
        )
        finger_pos.append(trainer.robot_controller.get_current_finger())

    # Save last frames to poses file
    save_last_frames(full_data_path, poses_data_path, valid_points, finger_pos)
    return poses_data_path

def generate_lift_poses(config):
    """
    Generate a grid that spans the entire object with cell_size x cell_size cells
    For each grid point, find the object point that is directly below it
    If such a point exists, generate a trajectory with InvPhyTrainerWarp.generate_data where the robot starts open, 0.1 units above the point, lowers itself to the point, closes, and stays stationary for {wait} frames
    Tell generate_data to save to PhysTwin/generated_data/{case_name}/full_lift_poses.h5
    Create a file PhysTwin/generated_data/{case_name}/lift_poses.h5 that contains the last frame of each trajectory
    """
    case_name = config["case_name"]
    cell_size = config["cell_size"]
    wait = config["wait"]
    
    print(f"Generating lift poses for case: {case_name}")
    
    # Initialize PhysTwin
    trainer, config, object_vertices = init_phystwin(case_name)
    
    # Generate grid points on the object surface
    print("Generating grid points on object surface...")
    
    # Calculate bounding box of object
    min_coords = torch.min(object_vertices, dim=0)[0]
    max_coords = torch.max(object_vertices, dim=0)[0]
    
    # Generate 2D grid on x-y plane
    x_range = torch.arange(min_coords[0], max_coords[0] + cell_size, cell_size)
    y_range = torch.arange(min_coords[1], max_coords[1] + cell_size, cell_size)
    
    valid_points = []
    finger_pos = []
    
    for x in x_range:
        for y in y_range:
            grid_point = torch.tensor([x, y], dtype=torch.float32, device=object_vertices.device)
            
            # Find object points directly below this grid point
            # Project object points to x-y plane and find closest ones
            distances = torch.norm(object_vertices[:, :2] - grid_point, dim=1)
            
            # Find points within cell_size/2 distance
            close_indices = torch.where(distances < cell_size / 3)[0]
            
            if len(close_indices) > 0:
                # Use the highest point (highest z coordinate) among close points
                close_points = object_vertices[close_indices]
                highest_z_idx = torch.argmax(close_points[:, 2])
                target_point = close_points[highest_z_idx]
                
                valid_points.append(target_point)
    
    print(f"Found {len(valid_points)} valid lift points")
    
    # Setup output directories
    output_dir = GENERATED_DATA_DIR / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_data_path = str(output_dir / "full_lift_poses.h5")
    poses_data_path = str(output_dir / "lift_poses.h5")
    
    # Initialize data files
    if os.path.exists(full_data_path):
        os.remove(full_data_path)
    if os.path.exists(poses_data_path):
        os.remove(poses_data_path)
    
    # Generate trajectories for each valid point
    for i, target_point in enumerate(valid_points):
        print(f"Processing lift point {i+1}/{len(valid_points)}: {target_point.cpu().numpy()}")
        
        # Generate trajectory
        trainer.generate_data(
            config.get_best_model_path(),
            create_lift_action(target_point, wait),
            config.get_gaussian_path(),
            n_ctrl_parts=1,
            data_file_path=full_data_path,
            episode_id=i
        )
        finger_pos.append(trainer.robot_controller.get_current_finger())
    
    # Save last frames to poses file
    save_last_frames(full_data_path, poses_data_path, valid_points, finger_pos)
    return poses_data_path
    
def visualize_poses(poses_data_path):
    """
    Visualize the poses in the poses file
    """
    import open3d as o3d
    with h5py.File(poses_data_path, 'r') as f:
        object = f['object'][0]
        targets = f['target'][:]

    # Visualize the poses
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object)
    object_pcd.paint_uniform_color([0, 1, 0])

    targets[:, 2] += -0.003 # slight offset to avoid z-fighting
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(targets)
    target_pcd.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([object_pcd, target_pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["push", "lift"], required=True)
    parser.add_argument("--margin", type=float, default=None, help="Margin around object for grid generation (push)")
    parser.add_argument("--cell_size", type=float, default=None, help="Size of grid cells")
    parser.add_argument("--max_dist", type=float, default=None, help="Maximum distance from object point (push)")
    parser.add_argument("--min_dist", type=float, default=None, help="Minimum distance from all object points (push)")
    parser.add_argument("--wait", type=int, default=None, help="Number of frames to wait for object to stabilize")
    args = parser.parse_args()

    from GNN.utils import load_yaml
    config = load_yaml("PhysTwin/data_generation/config.yaml")
    config = config["start_pose"]
    if args.case_name is not None:
        config["case_name"] = args.case_name
    if args.margin is not None:
        config["margin"] = args.margin
    if args.cell_size is not None:
        config["cell_size"] = args.cell_size
    if args.max_dist is not None:
        config["max_dist"] = args.max_dist
    if args.min_dist is not None:
        config["min_dist"] = args.min_dist
    if args.wait is not None:
        config["wait"] = args.wait
    
    if args.mode == "push":
        poses_data_path = generate_push_poses(config)
    elif args.mode == "lift":
        poses_data_path = generate_lift_poses(config)
    # poses_data_path = "PhysTwin/generated_data/single_push_rope/lift_poses.h5"

    visualize_poses(poses_data_path)
    