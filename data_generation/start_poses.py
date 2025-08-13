from ..qqtt import InvPhyTrainerWarp
from ..qqtt.utils import PhysTwinConfig
import argparse
import torch
import numpy as np
import os
import h5py
import pickle
from ..paths import GENERATED_DATA_DIR
from .save_data import *

def create_push_action(grid_point, wait_frames):
    """
    Create an action function that positions the robot at a specific grid point and stays stationary.
    
    Args:
        grid_point: torch.Tensor [3] - target position for robot center
        wait_frames: int - number of frames to wait stationary
        
    Returns:
        function: Action function compatible with InvPhyTrainerWarp.generate_traj_from_act_func
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

def create_lift_action(grid_point, config):
    """
    Create an action function that positions the robot above a grid point, lowers to grasp, and stays stationary.
    
    Args:
        grid_point: torch.Tensor [3] - target position for robot center
        wait_frames: int - number of frames to wait stationary after grasping
        
    Returns:
        function: Action function compatible with InvPhyTrainerWarp.generate_traj_from_act_func
    """
    grid_point[2] += -0.005 # offset between robot center and lower end
    def lift_action(init_vertices, robot_controller, n_ctrl_parts):
        approach_end = config["approach"] 
        grasp_end = approach_end + config["grasp"]
        lift_end = grasp_end + config["lift"]
        wait_end = lift_end + config["wait"]
        
        # Robot starts open (finger = 1.0)
        initial_finger = 1.0
        target_changes = torch.zeros((wait_end, n_ctrl_parts, 3), dtype=torch.float32, device=robot_controller.device)
        finger_changes = torch.zeros(wait_end, dtype=torch.float32, device=robot_controller.device)
        
        # Calculate initial translation to position robot above the point
        current_robot_center = robot_controller.get_current_center()

        # Position 0.1 units above the target point
        start_position = grid_point.clone()
        start_position[2] += config["offset"]
        initial_translation = start_position - current_robot_center
                
        # Approach: move down to the target point
        approach_speed = config["offset"] / config["approach"]
        target_changes[0:approach_end, 0, 2] = -approach_speed
        
        # Grasp: wait for gripper to close
        finger_changes[approach_end:grasp_end] = -0.05

        # Lift: lift sloth up
        target_changes[grasp_end:lift_end, 0, 2] = approach_speed
        
        # Wait: wait for object to stabilize
        # Do nothing
        
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
        pure_inference_mode=True,
        static_meshes=[],
        robot_controller=config.get_robot_controller("default", n_ctrl_parts=1, device='cuda'),
    )

    # Get initial vertices
    with open(config.get_data_path(), 'rb') as f:
        data = pickle.load(f)
    # print(data.keys())
    object_vertices = torch.tensor(data['object_points'][0], dtype=torch.float32, device='cuda')
    # print(object_vertices.shape)

    return trainer, config, object_vertices

def generate_push_poses(config):
    """
    Generate a grid that spans the entire object + margin with cell_size x cell_size cells
    For each grid point, determine if it's [min_dist, max_dist] away from the closest object point
    If it is, generate a trajectory where the robot starts closed and centered at the point
    Wait for the object to stablize
    Save full trajectory to PhysTwin/generated_data/{case_name}/full_push_poses.h5 (not used for training)
    Save last frame to PhysTwin/generated_data/{case_name}/push_poses.h5 as an initial state for generating training data
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
    margin = torch.tensor([margin, margin, -cell_size], device = object_vertices.device)
    min_coords = torch.min(object_vertices, dim=0)[0] - margin
    max_coords = torch.max(object_vertices, dim=0)[0] + margin
    
    # Generate grid
    x_range = torch.arange(min_coords[0], max_coords[0] + cell_size, cell_size)
    y_range = torch.arange(min_coords[1], max_coords[1] + cell_size, cell_size)
    if min_coords[2] < max_coords[2]:
        z_range = torch.arange(min_coords[2], max_coords[2] + cell_size, cell_size)
    else:
        # z_range is just the midpoint
        z_range = torch.tensor([(min_coords[2] + max_coords[2]) / 2], device=object_vertices.device)
    
    valid_points = []

    for x in x_range:
        for y in y_range:
            for z in z_range:
                grid_point = torch.tensor([x, y, z], dtype=torch.float32, device=object_vertices.device)
                
                # Check distance constraints
                distances = torch.norm(object_vertices[:, :2] - grid_point[:2], dim=1) # calculate distance in x-y plane to avoid initializing inside the object
                min_distance = torch.min(distances).item()
                    
                # Point is valid if it's [min_dist, max_dist] away from the closest object point
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
        object_data, robot_data, gaussians_data, finger_pos = trainer.generate_traj_from_act_func(
            config.get_best_model_path(),
            create_push_action(grid_point, wait),
            config.get_gaussian_path(),
            n_ctrl_parts=1
        )
        save_episode_data(full_data_path, i, object_data, robot_data, grid_point, finger_pos)

    # Save last frames to poses file
    save_last_frames(full_data_path, poses_data_path)
    save_first_states(case_name, poses_data_path)
    return poses_data_path

def generate_lift_poses(dg_config):
    """
    Generate a grid that spans the entire object with cell_size x cell_size cells
    For each grid point, find the object point that is directly below it
    If such a point exists, generate a trajectory where the robot starts open, 0.1 units above the point, lowers itself to the point, and closes
    Wait for the gripper to close and the object to stablize
    Save full trajectory to PhysTwin/generated_data/{case_name}/full_lift_poses.h5 (not used for training)
    Save last frame to PhysTwin/generated_data/{case_name}/lift_poses.h5 as an initial state for generating training data
    """
    case_name = dg_config["case_name"]
    cell_size = dg_config["cell_size"]
    
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
                grid_point = close_points[highest_z_idx]
                
                valid_points.append(grid_point)
    
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
    for i, grid_point in enumerate(valid_points):
        print(f"Processing lift point {i+1}/{len(valid_points)}: {grid_point.cpu().numpy()}")
        
        # Generate trajectory
        object_data, robot_data, gaussians_data, finger_pos = trainer.generate_traj_from_act_func(
            config.get_best_model_path(),
            create_lift_action(grid_point, dg_config),
            config.get_gaussian_path(),
            n_ctrl_parts=1,
        )
        save_episode_data(full_data_path, i, object_data, robot_data, grid_point, finger_pos)

    # Save last frames to poses file
    save_last_frames(full_data_path, poses_data_path)
    save_first_states(case_name, poses_data_path)
    return poses_data_path
    
def visualize_poses(poses_data_path):
    """
    Visualize the poses in the poses file and save the visualization as an image.
    """
    import open3d as o3d
    import numpy as np

    with h5py.File(poses_data_path, 'r') as f:
        object = f['object'][0]
        targets = f['target'][:]

    # Visualize the poses
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object)
    object_pcd.paint_uniform_color([0, 1, 0])

    save_path = os.path.splitext(poses_data_path)[0] + ".png"

    targets[:, 2] += -0.003 # slight offset to avoid z-fighting
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(targets)
    target_pcd.paint_uniform_color([1, 0, 0])

    # Create a visualizer and add geometries
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(object_pcd)
    vis.add_geometry(target_pcd)
    vis.poll_events()
    vis.update_renderer()

    # Capture and save the image
    img = vis.capture_screen_float_buffer(do_render=True)
    img_np = (255 * np.asarray(img)).astype(np.uint8)
    from PIL import Image
    Image.fromarray(img_np).save(save_path)
    print(f"Saved visualization to {save_path}")

    # Keep the window open for user to view
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["push", "lift"], required=True)
    args = parser.parse_args()

    from GNN.utils import load_yaml
    config = load_yaml("PhysTwin/data_generation/config.yaml")
    config = config["start_pose"]
    if args.case_name is not None:
        config["case_name"] = args.case_name
    
    if args.mode == "push":
        poses_data_path = generate_push_poses(config)
    elif args.mode == "lift":
        poses_data_path = generate_lift_poses(config)
    # poses_data_path = "PhysTwin/generated_data/single_push_rope/lift_poses.h5"

    visualize_poses(poses_data_path)
    