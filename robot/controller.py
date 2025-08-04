"""
Stateful robot pose and movement controller.
Refactored from RobotMovementController to be the single source of robot state truth.
"""
import torch
import numpy as np
import open3d as o3d


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
    
    def __init__(self, robot_loader, init_finger, n_ctrl_parts=1, device='cuda', num_substeps=667, dt=5e-5):
        """
        Initialize the robot controller.
        
        Args:
            robot_loader: RobotLoader instance for getting finger meshes
            init_finger: torch.Tensor [n_ctrl_parts] - initial finger opening value(s) [0.0, 1.0]
            n_ctrl_parts: Number of control parts (default: 1)
            device: PyTorch device
            num_substeps: Number of substeps for interpolation (default: 667)
            dt: Time step for simulation (default: 5e-5)
        """
        self.n_ctrl_parts = n_ctrl_parts
        self.device = device
        self.robot_loader = robot_loader
        self.num_substeps = num_substeps
        self.dt = dt
    
        self.accumulate_trans = torch.zeros((n_ctrl_parts, 3), dtype=torch.float32, device=device)
        self.origin_force_judge = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.float32, device=device
        )
        
        """Reset all state variables to initial values."""
        self.accumulate_trans.zero_()
        self.accumulate_rot = torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n_ctrl_parts, 3, 3)
        
        self.current_finger = torch.clamp(init_finger, 0.0, 1.0)
        self.is_closing = init_finger < 0.5  # True if finger is more than half closed
        self.close_flag = init_finger != 0.0 # True if finger isn't fully closed
        self.current_force_judge = self.origin_force_judge.clone().unsqueeze(0).expand(n_ctrl_parts, 2, 3)

        # Update finger meshes and vertices for current finger position
        self.finger_meshes = self.get_finger_meshes()
        assert len(self.finger_meshes) % n_ctrl_parts == 0, "Number of finger meshes must be divisible by n_ctrl_parts"
        self.n_links = len(self.finger_meshes) // n_ctrl_parts
        
        # Get all finger vertices and create flattened tensor
        finger_vertices = [np.asarray(mesh.vertices) for mesh in self.finger_meshes]
        base_finger_vertices_np = np.concatenate(finger_vertices, axis=0)
        self.n_particles = base_finger_vertices_np.shape[0] // n_ctrl_parts
        
        # Reshape to [n_ctrl_parts, n_particles, 3] and convert to tensor
        base_vertices_reshaped = base_finger_vertices_np.reshape(n_ctrl_parts, self.n_particles, 3)
        self.current_trans_dynamic_points = torch.tensor(base_vertices_reshaped, dtype=torch.float32, device=self.device)
        self.dynamic_points = self.current_trans_dynamic_points.clone()

    def get_finger_meshes(self):
        """
        Get finger meshes for all fingers.
        
        Returns:
            list of o3d.geometry.TriangleMesh
        """
        finger_meshes = []
        for i in range(self.n_ctrl_parts):
            finger_meshes = finger_meshes + self.robot_loader.get_finger_mesh(self.current_finger[i].item())
        return finger_meshes
    
    def _get_base_finger_vertices(self):
        """
        Get base finger vertices for all fingers.
        
        Returns:
            torch.Tensor: Base finger vertices [n_ctrl_parts, n_particles, 3]
        """
        # Rebuild finger meshes for all parts
        self.finger_meshes = self.get_finger_meshes()
        
        # Get all finger vertices and concatenate
        finger_vertices = [np.asarray(mesh.vertices) for mesh in self.finger_meshes]
        base_finger_vertices_np = np.concatenate(finger_vertices, axis=0)
        
        # Reshape to [n_ctrl_parts, n_particles, 3]
        base_vertices_reshaped = base_finger_vertices_np.reshape(self.n_ctrl_parts, self.n_particles, 3)
        return torch.tensor(base_vertices_reshaped, dtype=torch.float32, device=self.device)
    
    def get_flattened_dynamic_points(self):
        """
        Get dynamic points in flattened format for compatibility.
        
        Returns:
            torch.Tensor: dynamic_points [n_ctrl_parts * n_particles, 3]
        """
        return self.dynamic_points.reshape(-1, 3)
        
    def quick_robot_movement(self, target_change, current_finger=None, rot_change=None):
        """
        Quickly teleport robot to desired pose (for initialization).
        
        Args:
            target_change: torch.Tensor [n_ctrl_parts, 3] - translation change
            current_finger: torch.Tensor [n_ctrl_parts] - target finger opening value(s) [0.0, 1.0] (default: None, keep current)
            rot_change: torch.Tensor [n_ctrl_parts, 3] - rotation change as axis-angle (default: None)
            
        Returns:
            torch.Tensor: dynamic_points [n_ctrl_parts * n_particles, 3]
        """
        # Handle default rotation
        if rot_change is None:
            rot_change = torch.zeros((self.n_ctrl_parts, 3), dtype=torch.float32, device=self.device)
        
        # Update rotation for each part
        self.accumulate_rot = self.accumulate_rot @ axis_angle_to_matrix(rot_change)
        
        # Update internal states
        self.accumulate_trans += target_change
        
        # Handle finger updates
        if current_finger is not None:
            self.current_finger = torch.clamp(current_finger, 0.0, 1.0)
            self.is_closing = self.current_finger < 0.5
        
        # Update mesh vertices for all grippers
        self.current_trans_dynamic_points = self._get_base_finger_vertices()
        
        # Add translation to all particles of each gripper
        self.current_trans_dynamic_points += self.accumulate_trans.unsqueeze(1)  # [n_ctrl_parts, 1, 3]
        
        # Calculate centers for all grippers in batch
        centers = torch.mean(self.current_trans_dynamic_points, dim=1)  # [n_ctrl_parts, 3]
        
        centered_points = self.current_trans_dynamic_points - centers.unsqueeze(1)
        rotated_points = centered_points @ self.accumulate_rot.transpose(1, 2)
        self.dynamic_points = rotated_points + centers.unsqueeze(1)
 
        # Update force judge direction for each part
        self.current_force_judge = self.origin_force_judge.unsqueeze(0).expand(self.n_ctrl_parts, 2, 3) @ self.accumulate_rot
        
        return self.get_flattened_dynamic_points()
        
    def fine_robot_movement(self, target_change, collision_forces, finger_change=None, rot_change=None):
        """
        Smoothly move robot with interpolated motion (for simulation).
        
        Args:
            target_change: torch.Tensor [n_ctrl_parts, 3] - translation change for this step
            finger_change: torch.Tensor [n_ctrl_parts] - change in finger opening (default: None, auto-determine)
            rot_change: torch.Tensor [n_ctrl_parts, 3] or None - rotation change as axis-angle (default: None)
            
        Returns:
            dict containing:
                interpolated_dynamic_points: [num_substeps, n_ctrl_parts * n_particles, 3] - robot mesh points
                interpolated_center: [num_substeps, n_ctrl_parts, 3] - center points for each substep
                dynamic_velocity: [n_ctrl_parts, 3] - robot velocity
                dynamic_omega: [n_ctrl_parts, 3] - robot angular velocity
        """
        # Store previous state
        prev_accumulate_rot = self.accumulate_rot.clone()
        prev_trans_dynamic_points = self.current_trans_dynamic_points.clone()

        # Handle default rotation
        if rot_change is None:
            rot_change = torch.zeros((self.n_ctrl_parts, 3), dtype=torch.float32, device=self.device)
        
        # Handle finger_change
        if finger_change is None:
            finger_change = torch.zeros(self.n_ctrl_parts, dtype=torch.float32, device=self.device)

        close_flag = self.get_close_flag(collision_forces)
            
        # Update is_closing flag based on finger_change
        self.is_closing[finger_change > 0] = False
        self.is_closing[finger_change < 0] = True
        finger_change[self.is_closing & ~close_flag] = 0.0
        finger_change[self.is_closing & close_flag] = -0.05
        finger_change[~self.is_closing] = 0.05

        # Update translation
        self.accumulate_trans += target_change
        
        # Update finger values
        self.current_finger = torch.clamp(self.current_finger + finger_change, 0.0, 1.0)
        self.current_trans_dynamic_points = self._get_base_finger_vertices() + self.accumulate_trans.unsqueeze(1)  # [n_ctrl_parts, 1, 3]
        
        # Calculate interpolated points considering finger and translation
        ratios = (
            torch.linspace(1, self.num_substeps, self.num_substeps, device=self.device).view(-1, 1, 1, 1)
            / self.num_substeps
        )
        
        # Interpolate from previous to current mesh positions
        interpolated_trans_dynamic_points = (
            prev_trans_dynamic_points.unsqueeze(0)
            + (self.current_trans_dynamic_points - prev_trans_dynamic_points).unsqueeze(0)
            * ratios
        ) # [num_substeps, n_ctrl_parts, n_particles, 3]
        
        # Calculate centers for each gripper at each substep
        interpolated_center = torch.mean(interpolated_trans_dynamic_points, dim=2) # [num_substeps, n_ctrl_parts, 3]
        
        # Do the rotation on the interpolated points for each gripper
        interpolated_rot_angle = rot_change.unsqueeze(0) * ratios.reshape(-1, 1, 1)
        interpolated_rot_mat = prev_accumulate_rot.unsqueeze(0) @ axis_angle_to_matrix(interpolated_rot_angle)
        self.accumulate_rot = interpolated_rot_mat[-1]

        # Apply rotation about centers in batch
        
        # Apply progressive rotation about interpolated centers for each gripper
        centered_points = interpolated_trans_dynamic_points - interpolated_center.unsqueeze(2)
        rotated_points = torch.einsum('snpi,snij->snpj', centered_points, interpolated_rot_mat.transpose(2, 3))
        interpolated_dynamic_points = rotated_points + interpolated_center.unsqueeze(2)
        self.dynamic_points = interpolated_dynamic_points[-1]
        
        # Update force judge direction for each gripper
        self.current_force_judge = self.origin_force_judge.unsqueeze(0).expand(self.n_ctrl_parts, 2, 3) @ interpolated_rot_mat[-1]
        
        # Calculate velocity and omega for each gripper
        dynamic_velocity = target_change / (2 * self.dt * self.num_substeps)
        dynamic_omega = rot_change / (2 * self.dt * self.num_substeps)
        
        # Reshape interpolated_dynamic_points to flattened format for compatibility
        flattened_interpolated_points = interpolated_dynamic_points.reshape(self.num_substeps, -1, 3)
        
        return {
            'interpolated_dynamic_points': flattened_interpolated_points,
            'interpolated_center': interpolated_center,
            'dynamic_velocity': dynamic_velocity,
            'dynamic_omega': dynamic_omega
        }

    def get_close_flag(self, collision_forces):
        """
        Get close flag based on collision forces.
        
        Args:
            collision_forces: torch.Tensor [n_ctrl_parts * n_links, 3] - collision forces
        """
        collision_forces = collision_forces[:self.n_ctrl_parts * self.n_links]
        collision_forces = collision_forces.reshape(self.n_ctrl_parts, self.n_links, 3)
        filter_forces = torch.einsum(
            "nij,nij->ni", collision_forces, self.current_force_judge
        )
        return torch.all(filter_forces > 3e4, dim=1)

    def set_to_match_vertices(self, target_vertices):
        """
        Set robot state to match given mesh vertices by calculating the required translation.
        
        Args:
            target_vertices: torch.Tensor [n_ctrl_parts * n_particles, 3] - target mesh vertices
        """
        # Reshape target vertices to [n_ctrl_parts, n_particles, 3] if needed
        if target_vertices.dim() == 2:
            assert target_vertices.shape[0] == self.n_ctrl_parts * self.n_particles, "Target vertices dimension mismatch"
            target_vertices = target_vertices.reshape(self.n_ctrl_parts, self.n_particles, 3)
        
        # Calculate translation for each gripper
        current_mesh_centers = torch.mean(self.current_trans_dynamic_points, dim=1)  # [n_ctrl_parts, 3]
        target_mesh_centers = torch.mean(target_vertices, dim=1)  # [n_ctrl_parts, 3]
        translations = target_mesh_centers - current_mesh_centers
        
        self.quick_robot_movement(
            target_change=translations,
            current_finger=torch.zeros(self.n_ctrl_parts, dtype=torch.float32, device=self.device), # by default close the grippers
        )
                
    def get_current_state(self):
        """Get current robot state for serialization/debugging."""
        return {
            'accumulate_trans': self.accumulate_trans.clone(),
            'accumulate_rot': self.accumulate_rot.clone(),
            'current_finger': self.current_finger.copy(),
            'is_closing': self.is_closing.copy(),
            'dynamic_points': self.get_flattened_dynamic_points(),
            'current_force_judge': self.current_force_judge.clone()
        }
        
    def get_current_force_judge(self):
        """Get current force judge for collision detection."""
        return self.current_force_judge
        
    def get_current_finger(self):
        """Get current finger opening value(s)."""
        return self.current_finger
        
    def get_current_center(self):
        """Get current center of the robot for each part."""
        return torch.mean(self.current_trans_dynamic_points, dim=1)  # [n_ctrl_parts, 3]
