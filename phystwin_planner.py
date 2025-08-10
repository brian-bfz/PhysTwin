# Warning: this script hasn't been updated to reflect recent changes. 
import torch
import numpy as np
import os
import warp as wp
import open3d as o3d
import cv2
import pickle
import json

from .qqtt import InvPhyTrainerWarp
from .qqtt.utils import logger, cfg, PhysTwinConfig, create_common_parser
from .paths import get_case_paths, URDF_XARM7
from shared.planner import PlannerWrapper
from shared.utils import load_mpc_data, setup_task_directory
from GNN.utils import load_yaml
from shared.reward import RewardFn


class PhysTwinRolloutFn:
    def __init__(self, trainer : InvPhyTrainerWarp, robot_mask, device):
        """
        Initialize the PhysTwin model rollout function.
        
        Args:
            trainer: InvPhyTrainerWarp instance with loaded model
            robot_mask: [n_particles] - boolean tensor for robot particles
            device: torch device
        """
        import warp as wp
        self.trainer = trainer
        self.robot_mask = robot_mask
        self.device = device
        
    def __call__(self, state_cur, action_seqs):
        """
        Evaluate multiple action sequences using sequential PhysTwin rollouts.
        
        Args:
            state_cur: [1, state_dim] - current state (object + robot particles)
            action_seqs: [n_sample, n_look_ahead, 2] - robot translation sequences
            
        Returns:
            dict containing:
                state_seqs: [n_sample, n_look_ahead, state_dim] - predicted trajectories
        """
        n_sample, n_look_ahead, _ = action_seqs.shape
        
        # Extract initial state
        initial_state = state_cur[-1].view(-1, 3)  # [n_particles, 3]
        initial_object_state = initial_state[~self.robot_mask]  # [n_obj, 3]
        initial_robot_state = initial_state[self.robot_mask]  # [n_bot, 3]
        
        # Sequential rollout for each action sequence sample
        state_seqs = torch.zeros(n_sample, n_look_ahead, state_cur.shape[1], device=self.device)
        
        for i in range(n_sample):
            predicted_states = self.trainer.generate_traj_from_act_seq(
                initial_object_state,
                initial_robot_state, 
                action_seqs[i]  # [n_look_ahead, 2]
            )
            state_seqs[i] = predicted_states.flatten(start_dim=1)  # [n_look_ahead, n_particles * 3]
            
        return {'state_seqs': state_seqs}
    

class PhysTwinPlanner(PlannerWrapper):
    """Wrapper for PhysTwin-based model predictive control"""
    
    def __init__(self, mpc_config: dict, case_name: str, 
                 base_path: str = None, bg_img_path: str = None, gaussian_path: str = None):
        """
        Initialize the PhysTwin planner wrapper.
        
        Args:
            mpc_config: dict - MPC configuration dictionary
            case_name: Name of the case/experiment
            base_path: Base path for data (optional)
            bg_img_path: Path to background image (optional)
            gaussian_path: Path to gaussian output directory (optional)
        """
        super().__init__(mpc_config)
        
        # Initialize configuration
        self.model_config = PhysTwinConfig(
            case_name=case_name,
            base_path=base_path,
            bg_img_path=bg_img_path,
            gaussian_path=gaussian_path
        )

        # Initialize trainer
        self.trainer = self.initialize_model()
        print(f"PhysTwin MPC initialized for case: {case_name}")

    def initialize_model(self):
        """Initialize PhysTwin trainer with loaded model."""
        # Create trainer
        trainer = InvPhyTrainerWarp(
            data_path=self.model_config.get_data_path(),
            base_dir=str(self.model_config.case_paths['base_dir']),
            pure_inference_mode=True,
            static_meshes=[],
            robot_controller=self.model_config.get_robot_controller("default"),
        )
        
        # Initialize simulator with trained model
        best_model_path = self.model_config.get_best_model_path()
        trainer.initialize_simulator(best_model_path)
        
        return trainer

    def _prepare_initial_state(self, first_states, robot_mask):
        """Prepare initial state for PhysTwin model."""
        # PhysTwin doesn't need history - just current state
        state_cur = first_states.clone().unsqueeze(0)  # [1, n_particles, 3]
        state_cur = state_cur.flatten(start_dim=1)  # [1, n_particles * 3]
        return state_cur

    def _create_model_rollout_fn(self, robot_mask, first_states, topological_edges):
        """Create PhysTwin model rollout function."""
        return PhysTwinRolloutFn(self.trainer, robot_mask, self.device)

    def _create_reward_fn(self, robot_mask, target_pcd=None):
        """Creates the standard goal-oriented reward function for PhysTwin."""
        return RewardFn(self.action_weight, self.fsp_weight, robot_mask, target_pcd)


def visualize_action_phystwin(save_dir, file_name, case_name):
    """
    Visualize PhysTwin action sequence using saved results.
    
    Args:
        save_dir: str - directory containing saved results
        file_name: str - filename in format episode_{idx}_{timestamp}
        case_name: str - case name for camera parameters
        
    Returns:
        str - path to saved video
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config file from save_dir
    config_path = os.path.join(save_dir, "mpc_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_data = load_yaml(config_path)
    
    # Load target
    target_path = os.path.join(save_dir, "target.npz")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target file not found: {target_path}")
    target_data = np.load(target_path)
    target_pcd = torch.tensor(target_data['target'], dtype=torch.float32, device=device)
    
    # Load predicted states from bmo file
    bmo_path = os.path.join(save_dir, f"bmo_{file_name}.npz")
    if not os.path.exists(bmo_path):
        raise FileNotFoundError(f"BMO file not found: {bmo_path}")
    bmo_data = np.load(bmo_path)
    predicted_states = torch.tensor(bmo_data['state_seqs'], device=device)
    
    # Extract episode index from filename
    episode_idx = int(file_name.split('_')[1])
    
    # Load data using config paths
    first_states, robot_mask, _ = load_mpc_data(episode_idx, config_data['data_file'], device)
    
    n_particles = len(robot_mask)
    
    # Reshape predicted states: [n_look_ahead, n_particles*3] -> [n_look_ahead, n_particles, 3]
    predicted_states = predicted_states.reshape(-1, n_particles, 3)
    
    # Add initial state to create full trajectory
    initial_state = first_states.unsqueeze(0)  # [1, n_particles, 3]
    full_trajectory = torch.cat([initial_state, predicted_states], dim=0)  # [n_look_ahead+1, n_particles, 3]
    
    # Load camera parameters from config files
    case_paths = get_case_paths(case_name)
    data_path = case_paths['data_dir']
    
    # Load camera calibration
    with open(f"{data_path}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    
    # Load metadata for intrinsics and window size
    with open(f"{data_path}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsics = np.array(data["intrinsics"])
    width, height = data["WH"]
    
    # Use first camera view
    vis_cam_idx = 0
    intrinsic = intrinsics[vis_cam_idx]
    w2c = w2cs[vis_cam_idx]
    
    # Setup visualization with proper camera parameters
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    render_option = vis.get_render_option()
    render_option.point_size = 10.0
    
    # Create point clouds
    object_pcd = o3d.geometry.PointCloud()
    robot_pcd = o3d.geometry.PointCloud()
    target_pcd_vis = o3d.geometry.PointCloud()
    
    # Set target point cloud (static, orange)
    target_pcd_vis.points = o3d.utility.Vector3dVector(target_pcd.cpu().numpy())
    target_pcd_vis.paint_uniform_color([1.0, 0.6, 0.2])  # Orange
    vis.add_geometry(target_pcd_vis)
    
    # Initialize with first frame
    initial_frame = full_trajectory[0].cpu().numpy()
    object_points = initial_frame[~robot_mask.cpu()]
    robot_points = initial_frame[robot_mask.cpu()]
    
    object_pcd.points = o3d.utility.Vector3dVector(object_points)
    object_pcd.paint_uniform_color([0, 1, 0])  # Green for predicted object states
    vis.add_geometry(object_pcd)
    
    robot_pcd.points = o3d.utility.Vector3dVector(robot_points)
    robot_pcd.paint_uniform_color([1, 0, 0])  # Red for robot
    vis.add_geometry(robot_pcd)
    
    # Setup camera view
    view_control = vis.get_view_control()
    camera_params = o3d.camera.PinholeCameraParameters()
    intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic)
    camera_params.intrinsic = intrinsic_parameter
    camera_params.extrinsic = w2c
    view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    
    # Create output video path
    video_path = os.path.join(save_dir, f"{file_name}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = cfg.FPS  # Use FPS from config
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Animate through trajectory
    for frame_idx in range(len(full_trajectory)):
        current_frame = full_trajectory[frame_idx].cpu().numpy()
        object_points = current_frame[~robot_mask.cpu()]
        robot_points = current_frame[robot_mask.cpu()]
        
        # Update point clouds
        object_pcd.points = o3d.utility.Vector3dVector(object_points)
        robot_pcd.points = o3d.utility.Vector3dVector(robot_points)
        
        vis.update_geometry(object_pcd)
        vis.update_geometry(robot_pcd)
        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame
        image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)
        
        # Show progress
        if frame_idx % 5 == 0:
            print(f"Rendering frame {frame_idx}/{len(full_trajectory)-1}")
    
    # Clean up
    out.release()
    vis.destroy_window()
    
    print(f"PhysTwin MPC visualization saved to: {video_path}")
    return video_path


if __name__ == "__main__":
    """
    Demonstrate PhysTwin-based model predictive control.
    """
    # Create parser with common arguments
    parser = create_common_parser()
    
    # Add script-specific arguments
    parser.add_argument("--episode", type=int, required=True,
                       help="Episode index to load")
    parser.add_argument("--mpc_config", type=str, default="PhysTwin/configs/mpc_config.yaml",
                       help="Path to MPC config file")
    parser.add_argument("--dir_name", type=str, required=True,
                       help="Directory name within tasks folder to store results")
    args = parser.parse_args()
    
    with torch.no_grad():
        # Load MPC config as dictionary
        mpc_config = load_yaml(args.mpc_config)
        
        # Initialize planner wrapper
        planner_wrapper = PhysTwinPlanner(
            mpc_config=mpc_config,
            case_name=args.case_name,
            base_path=args.base_path,
            bg_img_path=args.bg_img_path,
            gaussian_path=args.gaussian_path
        )

        # Load initial data to get robot mask for target creation
        first_states, robot_mask, topological_edges = load_mpc_data(args.episode, planner_wrapper.mpc_config['data_file'], planner_wrapper.device)
        
        # Setup task directory and get target
        save_dir, target_pcd = setup_task_directory(args.dir_name, planner_wrapper.mpc_config, planner_wrapper.device, model_type="PhysTwin")
    
        # Demonstrate planning
        print("="*60)
        print("PHYSTWIN-BASED MODEL PREDICTIVE CONTROL DEMO")
        print("="*60)
    
        # Plan action sequence for the specified episode
        planning_result = planner_wrapper.plan_action(target_pcd, first_states, robot_mask, topological_edges)
        planner_wrapper.save_planning_results(planning_result, args.episode, save_dir)
        optimal_actions = planning_result['act_seq']
        print(f"Planned action sequence shape: {optimal_actions.shape}")
        print("Planning demonstration complete!")

        # Visualize action sequence
        if 'best_model_output' in planning_result:
            # Generate filename from episode and timestamp
            timestamp = planner_wrapper.timestamp
            file_name = f"episode_{args.episode:06d}_{timestamp}"
            
            visualize_action_phystwin(save_dir, file_name, args.case_name) 