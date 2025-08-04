from PhysTwin.qqtt import InvPhyTrainerWarp
from PhysTwin.config_manager import PhysTwinConfig
from shared.reward import RewardFn
import torch
import torch.multiprocessing as mp
import h5py
import numpy as np
import os
from ..paths import *

class PhysTwin:
    """
    Compute ground truth deformation using PhysTwin.
    """
    
    def __init__(self, case_name, downsample_rate=1, device=None):
        """
        Initialize PhysTwin simulator.
        
        Args:
            case_name: str - case name for PhysTwin configuration
            downsample_rate: int - downsampling rate for configuration adjustment
            device: torch.device - device to use for computation (optional)
        """
        assert downsample_rate > 0 and isinstance(downsample_rate, int), "Downsample rate must be a positive integer"
        self.case_name = case_name
        self.downsample_rate = downsample_rate
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
                
        # Create PhysTwin configuration
        phystwin_config = PhysTwinConfig(case_name=self.case_name)
        
        # Keep PhysTwin at original frame rate - no adjustments needed
        # print(f"PhysTwin config: substeps={cfg.num_substeps}, FPS={cfg.FPS}")
        # print(f"GNN downsample_rate: {self.downsample_rate}")
        
        self.trainer = InvPhyTrainerWarp(
            data_path=phystwin_config.get_data_path(),
            base_dir=str(phystwin_config.case_paths['base_dir']),
            pure_inference_mode=True,
            static_meshes=[],
            robot_controller=phystwin_config.get_robot_controller(robot_type="default", device=self.device),
            device=str(self.device),
        )
        
        # Initialize simulator with trained model
        best_model_path = phystwin_config.get_best_model_path()
        self.trainer.initialize_simulator(best_model_path)
        
        # print("PhysTwin initialization completed!")
        
    def compute_deformation(self, initial_object_state, initial_robot_state, action_seq, init_finger=0.0):
        """
        Compute actual object deformation using PhysTwin simulation.
        
        This method interpolates the GNN action sequence to match PhysTwin's native frame rate,
        runs the simulation at full temporal resolution, then downsamples the results back to GNN frame rate.
        
        Args:
            action_seq: [n_look_ahead, action_dim] - action sequence at GNN frame rate
            initial_object_state: [n_obj, 3] - initial object states
            initial_robot_state: [n_bot, 3] - initial robot states
            init_finger: float - initial finger position
            
        Returns:
            actual_trajectory: [n_look_ahead*downsample_rate+1, n_particles, 3] - actual deformation trajectory at GNN frame rate
            downsampled_indices: [n_look_ahead] - indices of the downsampled frames
        """
        
        # Interpolate action sequence to match PhysTwin's native frame rate
        # GNN operates at downsampled rate, PhysTwin at original rate
        if self.downsample_rate > 1:
            interpolated_actions = (action_seq / self.downsample_rate).repeat_interleave(self.downsample_rate, dim=0)  # [n_look_ahead * downsample_rate, 2]
        else:
            interpolated_actions = action_seq
            
        # Get actual deformation from PhysTwin using rollout_act_seq
        # print("Computing actual deformation with PhysTwin...")
        predicted_states = self.trainer.rollout_act_seq(
            initial_object_state, initial_robot_state, interpolated_actions, init_finger
        )  # [n_look_ahead * downsample_rate, n_particles, 3]
            
        # Downsample predicted states back to GNN frame rate
        initial_state = torch.cat([initial_object_state, initial_robot_state], dim=0).unsqueeze(0) # [1, n_particles, 3]
        actual_trajectory = torch.cat([initial_state, predicted_states], dim=0)  # [n_look_ahead*downsample_rate+1, n_particles, 3]
        downsampled_indices = torch.arange(0, actual_trajectory.shape[0], self.downsample_rate, device=self.device)
        # print(f"Downsampled indices: {downsampled_indices}")
                        
        return actual_trajectory, downsampled_indices
            
def random_direction_3d(device='cpu'):
    direction = torch.randn(3, device=device)
    direction = direction / torch.norm(direction)
    return direction

def push_act_seq(config, object_vertices, robot_vertices):
    """
    Generate a random pushing action sequence for 2D robot movements.
    
    Args:
        config: dict containing:
            speed: dict containing:
                min: float - minimum speed
                max: float - maximum speed
            pause_duration: dict containing:
                min: int - minimum pause duration
                max: int - maximum pause duration
            move_duration: dict containing:
                min: int - minimum move duration
                max: int - maximum move duration
            n_frames: int - number of frames in the sequence
            p_random: float - probability that the robot moves in a random direction instead of toward the object
        object_vertices: [n_obj_particles, 3] - object vertex positions
        robot_vertices: [n_robot_particles, 3] - robot vertex positions  
        
    Returns:
        act_seq: [n_frames, 2] - robot velocity sequence (x, y components only)
    """
    from shared.utils import random_direction
    min_speed = config["speed"]["min"]
    max_speed = config["speed"]["max"]
    pause = config["pause_duration"]
    move = config["move_duration"]
    n_frames = config["n_frames"]
    p_random = config["p_random"]
    device = object_vertices.device
    
    # select a direction to move toward
    if torch.rand(1).item() < p_random:
        rd = random_direction(device)[:2] # [2] tensor
    else: # 80% chance to move toward a randomly selected object point
        random_idx = torch.randint(0, object_vertices.shape[0], (1,)).item()
        target_point = object_vertices[random_idx][:2]
        robot_center = robot_vertices.mean(dim=0)[:2]
        rd = (target_point - robot_center) / torch.norm(target_point - robot_center) # [2] tensor

    # Execute pushing motion
    current_frame = 0
    act_seq = torch.zeros((n_frames, 2), dtype=torch.float32, device=object_vertices.device)
    
    while current_frame < n_frames:
        move_duration = torch.randint(move["min"], move["max"], (1,)).item()
        move_duration = min(move_duration, n_frames - current_frame)

        speed = torch.rand(1).item() * (max_speed - min_speed) + min_speed
        
        act_seq[current_frame:current_frame + move_duration] = (speed * rd).unsqueeze(0)
        
        current_frame += move_duration
        
        pause_duration = torch.randint(pause["min"], pause["max"], (1,)).item()
        current_frame += pause_duration

    return act_seq

def lift_act_seq(config, device='cpu'):
    """
    Generate a random sequence of 3D movements and pauses
    Cumulative z displacement is always non-negative
    Args: 
        n_frames: int - number of frames in the sequence
        device: str or torch.device - device to create tensors on (default: 'cpu')
        
    Returns:
        act_seq: [n_frames, 3] - robot velocity sequence
    """
    min_speed = config["speed"]["min"]
    max_speed = config["speed"]["max"]
    n_frames = config["n_frames"]
    pause = config["pause_duration"]
    move = config["move_duration"]
    
    # Initialize action sequence
    act_seq = torch.zeros((n_frames, 3), dtype=torch.float32, device=device)
    current_frame = 0
    cumulative_z = 0.0
    
    while current_frame < n_frames:
        # Generate movement phase
        move_duration = torch.randint(move["min"], move["max"], (1,)).item()
        move_duration = min(move_duration, n_frames - current_frame)
        
        rd = random_direction_3d(device)
        speed = torch.rand(1).item() * (max_speed - min_speed) + min_speed
        
        min_z = 0 - cumulative_z / move_duration / speed
        assert min_z < 1e-9, f"min_z has to be negative, but got {min_z}"
        if rd[2].item() < min_z:
            rd[2] = -rd[2] # force the robot to move upward
        
        act_seq[current_frame:current_frame + move_duration] = (rd * speed).unsqueeze(0)
        current_frame += move_duration
        cumulative_z += rd[2].item() * speed * move_duration
        # print(cumulative_z)
        
        # Add pause between movements
        pause_duration = torch.randint(pause["min"], pause["max"], (1,)).item()
        current_frame += pause_duration
    
    act_seq[:, 2] = -act_seq[:, 2] # z axis is inverted
    return act_seq

def generate_data(args):
    """
    Generate training data using PhysTwin simulation for a given case.
    
    Args:
        args: tuple - (case_name, n_episodes, n_frames, output_file, mode, rank)
            case_name: str - case name for PhysTwin configuration
            n_episodes: int - number of episodes to generate
            n_frames: int - number of frames per episode
            output_file: str - output file name
            mode: str - "push" or "lift" action mode
            rank: int - GPU rank for multi-GPU processing (None for single GPU)
    Return:
        output_file: str - output file full path
    """
    torch.set_grad_enabled(False)
    config, n_episodes, output_file, mode, rank = args
    case_name = config["case_name"]
    n_frames = config["n_frames"]

    if rank is not None: 
        device = f'cuda:{rank}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    phystwin = PhysTwin(case_name, device=device)

    if mode == "push":
        input_file = f"PhysTwin/generated_data/{case_name}/push_poses.h5"
    else:
        input_file = f"PhysTwin/generated_data/{case_name}/lift_poses.h5"
    assert os.path.exists(input_file), f"{input_file} does not exist. Please run generate_start_poses first."

    if rank is not None: 
        output_file = GENERATED_DATA_DIR / f"{output_file}_{rank}.h5"
    else:
        output_file = GENERATED_DATA_DIR / f"{output_file}.h5"

    with h5py.File(input_file, 'r', swmr=True) as f_in:
        objects = torch.from_numpy(f_in['object'][:]).to(device)
        robots = torch.from_numpy(f_in['robot'][:]).to(device)
        if 'finger' in f_in:
            fingers = f_in['finger'][:]
        elif mode == "push":
            fingers = np.zeros(objects.shape[0])
        else:
            raise ValueError("Lift mode must have finger data")

    # Progress bar setup
    pbar = None
    if rank is None or rank == 0:
        from tqdm import tqdm
        pbar = tqdm(total=n_episodes, desc="Generating episodes")

    with h5py.File(output_file, 'w') as f_out:
        for i in range(n_episodes):
            # randomly select a start pose
            random_idx = torch.randint(0, objects.shape[0], (1,)).item()
            object_vertices = objects[random_idx]
            robot_vertices = robots[random_idx]
            finger = fingers[random_idx]
            n_obj = object_vertices.shape[0]
            n_bot = robot_vertices.shape[0]

            # Generate action sequence
            if mode == "push":
                act_seq = push_act_seq(config, object_vertices, robot_vertices)
            else:
                act_seq = lift_act_seq(config, device)
                    
            # Compute trajectory with PhysTwin
            trajectory, _ = phystwin.compute_deformation(object_vertices, robot_vertices, act_seq, init_finger=finger)
                
            # save the trajectory
            trajectory = trajectory.cpu().numpy()
            episode_group = f_out.create_group(f'episode_{i:06d}')
            episode_group.create_dataset('object', data=trajectory[:, :n_obj, :])
            episode_group.create_dataset('robot', data=trajectory[:, n_obj:, :])
            episode_group.attrs['n_frames'] = n_frames
            episode_group.attrs['n_obj_particles'] = n_obj
            episode_group.attrs['n_bot_particles'] = n_bot
            episode_group.attrs['case_name'] = case_name
            episode_group.attrs['mode'] = mode

            if pbar is not None:
                pbar.update(1)
    
    if pbar is not None:
        pbar.close()
    
    return output_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, required=True)
    parser.add_argument("--n_frames", type=int, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["push", "lift"], required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    from GNN.utils import load_yaml
    config = load_yaml("PhysTwin/data_generation/config.yaml")
    config = config["trajectory"]
    if args.n_frames is not None:
        config["n_frames"] = args.n_frames
    if args.case_name is not None:
        config["case_name"] = args.case_name
    
    mp.set_start_method('spawn')
    world_size = torch.cuda.device_count()
    if world_size > 1:
        with mp.Pool(world_size) as pool:
            output_files = pool.map(generate_data, [
                (
                    config, 
                    args.n_episodes // world_size + (1 if args.n_episodes % world_size > i else 0),
                    args.output_file, 
                    args.mode, 
                    i
                ) for i in range(world_size)
            ])
    else:
        output_file = generate_data((config, args.output_file, args.mode, None))

    if world_size > 1:
        from ..scripts.merge_dataset import merge_datasets
        output_file = GENERATED_DATA_DIR / f"{args.output_file}.h5"
        merge_datasets(output_files, output_file)
        for file in output_files:
            os.remove(file)

    if args.video:
        from ..visualize_data import video_from_data
        
        robot = PhysTwinConfig(case_name=config["case_name"]).get_robot_controller(device='cpu')
        meshes = robot.finger_meshes
        with h5py.File(output_file, 'r') as f:
            for i in range(args.n_episodes):
                video_from_data(f, i, meshes, str(GENERATED_VIDEOS_DIR / args.output_file))