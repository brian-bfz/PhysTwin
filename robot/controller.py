"""
Stateful robot pose and movement controller.
Refactored from RobotMovementController to be the single source of robot state truth.
"""
import torch
import numpy as np


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.
    
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
    cross_product_matrix = torch.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(shape + (3,))
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = torch.eye(3, dtype=dtype, device=device)
    angles_sqrd = angles * angles
    angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
    return (
        identity.expand(cross_product_matrix.shape)
        + torch.sinc(angles / torch.pi) * cross_product_matrix
        + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )


class RobotController:
    """
    Handles robot state management and movement control.
    
    This class is the single source of truth for robot state including:
    - Position and rotation
    - Gripper opening
    - Mesh vertices
    
    All internal computations use PyTorch tensors for efficiency.
    Only converts to NumPy at interface boundaries (Open3D, file I/O).
    """
    
    def __init__(self, robot_loader, init_finger, n_ctrl_parts=1, device='cuda'):
        """
        Initialize the robot controller.
        
        Args:
            robot_loader: RobotLoader instance for getting finger meshes
            n_ctrl_parts: Number of control parts (default: 1)
            device: PyTorch device
        """
        self.n_ctrl_parts = n_ctrl_parts
        self.device = device
        self.robot_loader = robot_loader
    
        self.accumulate_trans = torch.zeros((n_ctrl_parts, 3), dtype=torch.float32, device=device)
        self.origin_force_judge = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.float32, device=device
        )
        
        """Reset all state variables to initial values."""
        self.accumulate_trans.zero_()
        self.accumulate_rot = torch.eye(3, dtype=torch.float32, device=self.device)
        self.is_closing = True
        self.current_force_judge = self.origin_force_judge.clone()

        # Update finger meshes and vertices for current finger position
        self.current_finger = max(0.0, min(1.0, init_finger))
        self.finger_meshes = self.robot_loader.get_finger_mesh(self.current_finger)
        
        finger_vertices = [np.asarray(mesh.vertices) for mesh in self.finger_meshes]
        base_finger_vertices_np = np.concatenate(finger_vertices, axis=0)
        self.current_trans_dynamic_points = torch.tensor(base_finger_vertices_np, dtype=torch.float32, device=self.device)
        self.dynamic_points = self.current_trans_dynamic_points.clone()
        self.num_dynamic = self.dynamic_points.shape[0]

    def get_base_finger_vertices(self):
        self.finger_meshes = self.robot_loader.get_finger_mesh(self.current_finger)
        finger_vertices = [np.asarray(mesh.vertices) for mesh in self.finger_meshes]
        base_finger_vertices_np = np.concatenate(finger_vertices, axis=0)
        return torch.tensor(base_finger_vertices_np, dtype=torch.float32, device=self.device)
    
    def get_close_flag(self, collision_forces):
        if collision_forces is None:
            return False # if we don't know the collision forces, we assume collision
        collision_forces = collision_forces[:self.num_dynamic]
        filter_forces = torch.einsum(
            "ij,ij->i", collision_forces, self.current_force_judge
        )
        return not torch.all(filter_forces > 3e4)
        
    def quick_robot_movement(self, target_change, current_finger=0.0, rot_change=None):
        """
        Quickly teleport robot to desired pose (for initialization).
        
        Args:
            target_change: torch.Tensor [n_ctrl_parts, 3] - translation change
            current_finger: float - target finger opening value [0.0, 1.0]
            rot_change: torch.Tensor [3] - rotation change as axis-angle (default: None)
            
        Returns:
            torch.Tensor: dynamic_points [n_vertices, 3]
        """
        # Handle default rotation
        if rot_change is None:
            rot_change = torch.zeros(3, dtype=torch.float32, device=self.device)
            
        # Update rotation
        if torch.norm(rot_change) > 0:
            rot_mat = axis_angle_to_matrix(rot_change.unsqueeze(0))[0]
            self.accumulate_rot = torch.matmul(self.accumulate_rot, rot_mat)
        
        # Update internal states
        self.accumulate_trans += target_change
        self.current_finger = max(0.0, min(1.0, current_finger))
        self.current_trans_dynamic_points = self.get_base_finger_vertices() + self.accumulate_trans[0]
        current_center = self.get_current_center()
        self.dynamic_points = (self.current_trans_dynamic_points - current_center) @ self.accumulate_rot.T + current_center
        
        self.is_closing = False if self.current_finger >= 0.9 else True
 
        # Update force judge direction
        self.current_force_judge = self.origin_force_judge.clone() @ self.accumulate_rot.T
        
        return self.dynamic_points
        
    def fine_robot_movement(self, target_change, collision_forces=None, finger_change=0.0, rot_change=None):
        """
        Smoothly move robot with interpolated motion (for simulation).
        
        Args:
            target_change: torch.Tensor [n_ctrl_parts, 3] - translation change for this step
            collision_forces: torch.Tensor [num_dynamic, 3] - collision forces for each dynamic point
            finger_change: float - change in finger opening (default: 0.0)
            rot_change: torch.Tensor [3] - rotation change as axis-angle (default: None)
            
        Returns:
            dict containing:
                interpolated_dynamic_points: [num_substeps, n_robot_vertices, 3] - robot mesh points
                interpolated_center: [num_substeps, 3] - center points for each substep
                dynamic_velocity: [3] - robot velocity
                dynamic_omega: [3] - robot angular velocity
        """
        # Store previous state
        prev_accumulate_rot = self.accumulate_rot.clone()
        prev_trans_dynamic_points = self.current_trans_dynamic_points.clone()

        # Import cfg here to avoid circular imports
        from ..qqtt.utils import cfg
        
        # Handle default rotation
        if rot_change is None:
            rot_change = torch.zeros(3, dtype=torch.float32, device=self.device)
        
        close_flag = self.get_close_flag(collision_forces)
        
        # Update is_closing flag based on finger_change
        if finger_change > 0:
            self.is_closing = False
        elif finger_change < 0:
            self.is_closing = True

        if self.is_closing:
            if not close_flag:
                finger_change = 0.0
            else: 
                finger_change = -0.05
        else:
            finger_change = 0.05
            
        # Update translation
        self.accumulate_trans += target_change
        
        self.current_finger = max(0.0, min(1.0, self.current_finger + finger_change))
        if finger_change != 0:
            self.current_trans_dynamic_points = self.get_base_finger_vertices() + self.accumulate_trans[0]
        else:
            self.current_trans_dynamic_points += target_change
        
        # Calculate interpolated points considering finger and translation
        ratios = (
            torch.linspace(1, cfg.num_substeps, cfg.num_substeps, device=self.device).view(-1, 1, 1)
            / cfg.num_substeps
        )
        
        # Interpolate from previous to current mesh positions
        interpolated_trans_dynamic_points = (
            prev_trans_dynamic_points.unsqueeze(0)
            + (self.current_trans_dynamic_points - prev_trans_dynamic_points).unsqueeze(0)
            * ratios
        )
        interpolated_center = torch.mean(interpolated_trans_dynamic_points, dim=1)
        
        # Do the rotation on the interpolated points
        interpolated_rot_angle = rot_change.unsqueeze(0) * ratios.reshape(-1, 1)
        interpolated_rot_mat = prev_accumulate_rot.unsqueeze(0) @ axis_angle_to_matrix(interpolated_rot_angle)
        self.accumulate_rot = interpolated_rot_mat[-1]
        
        # Apply progressive rotation about interpolated centers
        interpolated_dynamic_points = (
            interpolated_trans_dynamic_points - interpolated_center.unsqueeze(1)
        ) @ interpolated_rot_mat.permute(0, 2, 1) + interpolated_center.unsqueeze(1)
        
        # Update final mesh vertices
        self.dynamic_points = interpolated_dynamic_points[-1]
        
        # Update force judge direction
        self.current_force_judge = self.origin_force_judge.clone() @ interpolated_rot_mat[-1].T
        
        # Calculate velocity and omega
        dynamic_velocity = target_change[0] / (2 * cfg.dt * cfg.num_substeps)
        dynamic_omega = rot_change / (2 * cfg.dt * cfg.num_substeps)
        
        return {
            'interpolated_dynamic_points': interpolated_dynamic_points,
            'interpolated_center': interpolated_center,
            'dynamic_velocity': dynamic_velocity,
            'dynamic_omega': dynamic_omega
        }
        
    def set_to_match_vertices(self, target_vertices, init_finger=0.0):
        """
        Set robot state to match given mesh vertices by calculating the required translation.
        
        Args:
            target_vertices: torch.Tensor [n_vertices, 3] - target mesh vertices
            init_finger: float [0,1] - initial finger opening value
        """
        current_mesh_center = self.get_current_center()
        target_mesh_center = torch.mean(target_vertices, dim=0)
        
        # Calculate translation from current to target
        translation = (target_mesh_center - current_mesh_center).unsqueeze(0)  # Shape: [1, 3] for n_ctrl_parts=1
        
        self.quick_robot_movement(
            target_change=translation,
            current_finger=init_finger,
            rot_change=None
        )
                
    def get_current_state(self):
        """Get current robot state for serialization/debugging."""
        return {
            'accumulate_trans': self.accumulate_trans.clone(),
            'accumulate_rot': self.accumulate_rot.clone(),
            'current_finger': self.current_finger,
            'is_closing': self.is_closing,
            'dynamic_points': self.dynamic_points.clone(),
            'current_force_judge': self.current_force_judge.clone()
        }
        
    def get_current_finger(self):
        """Get current finger opening value."""
        return self.current_finger
        
    def get_current_center(self):
        """Get current center of the robot."""
        return torch.mean(self.current_trans_dynamic_points, dim=0)
