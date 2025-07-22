"""
Action sequence generation functions for PhysTwin data generation.

This module contains standalone functions for generating different types of action sequences
that can be used for robotic manipulation data collection.
"""

import torch
import numpy as np
from shared.utils import random_direction

# DEPRECATED
def push_rope_once(init_vertices, robot_controller, n_ctrl_parts, offset_dist, keep_off, multiplier, speed):
    """
    Select a point on the rope and move pusher towards it.
    
    Args:
        init_vertices: Initial vertex positions
        robot_controller: Robot controller instance
        n_ctrl_parts: Number of control parts
        offset_dist: Distance to offset from the selected object point
        keep_off: Minimum distance to keep from all object points
        multiplier: Multiplier for the target movement distance
        speed: Speed of the robot movement
        
    Returns:
        tuple: (current_pose, target_changes)
            - current_pose: numpy array [4,4] for robot transformation matrix
            - target_changes: numpy array for target movement
    """
    max_attempts = 100
    for _ in range(max_attempts):
        # 1. Select a random point on the object
        random_idx = torch.randint(0, len(init_vertices), (1,)).item()
        selected_point = init_vertices[random_idx]
        
        # 2. Select a random direction
        rd = random_direction(init_vertices.device)
        
        # 3. Set starting position with offset
        start_position = selected_point + rd * offset_dist
        
        distances = torch.norm(init_vertices - start_position, dim=1)
        min_distance = torch.min(distances).item()
        
        if min_distance >= keep_off:
            # Valid position found
            break
    else:
        # If we couldn't find a valid position after max attempts
        raise RuntimeError("Could not find valid robot position after maximum attempts")
    
    # 5. Find robot's current position and calculate translation
    robot_vertices = robot_controller.dynamic_points.cpu().numpy()
    robot_position = np.mean(robot_vertices, axis=0)
    translation = start_position.cpu().numpy() - robot_position
    translation = translation.reshape(n_ctrl_parts, 3)
    
    # 6. Calculate total movement distance and number of frames needed
    movement_vector = -random_direction(init_vertices.device).cpu().numpy() * offset_dist * multiplier
    total_distance = np.linalg.norm(movement_vector)
    n_frames = max(1, int(np.ceil(total_distance / speed)))
    
    # 7. Divide the movement into n_frames steps
    step_movement = movement_vector / n_frames
    target_changes = np.zeros((n_frames, n_ctrl_parts, 3))
    for i in range(n_frames):
        target_changes[i, 0] = step_movement
            
    return translation.astype(np.float32), target_changes.astype(np.float32)


def push_once(init_vertices, robot_controller, n_ctrl_parts):
    """
    Select a point on the rope, offset it by offset_dist, and move the pusher towards it (75% chance), 
    or in a random direction (25% chance)
    
    Args:
        init_vertices: Initial vertex positions
        robot_controller: Robot controller instance
        n_ctrl_parts: Number of control parts
        
    Returns:
        tuple: (current_pose, target_changes, initial_finger, finger_changes)
            - current_pose: numpy array [4,4] for robot transformation matrix
            - target_changes: numpy array for target movement
            - initial_finger: initial finger position
            - finger_changes: finger movement changes
    """
    offset_dist = 0.08
    keep_off = 0.03
    n_frames = 121
    min_speed = 0.004

    # finger is always closed
    initial_finger = 0.0
    finger_changes = torch.zeros(n_frames, dtype=torch.float32, device=robot_controller.device)
    
    # Get object points from the simulator
    max_attempts = 100
    for _ in range(max_attempts):
        # 1. Select a random point on the object
        random_idx = torch.randint(0, len(init_vertices), (1,)).item()
        selected_point = init_vertices[random_idx]
        
        # 2. Select a random direction
        rd = random_direction(init_vertices.device)
        
        # 3. Set starting position with offset
        start_position = selected_point + rd * offset_dist
        
        distances = torch.norm(init_vertices - start_position, dim=1)
        min_distance = torch.min(distances).item()
        
        if min_distance >= keep_off:
            # Valid position found
            break
    else:
        # If we couldn't find a valid position after max attempts
        raise RuntimeError("Could not find valid robot position after maximum attempts")
    
    # Calculate the initial translation to send the robot center to the selected point
    current_robot_center = robot_controller.get_current_center()
    initial_translation = start_position - current_robot_center

    # 5. Select a new random direction
    if torch.rand(1).item() < 0.4:
        rd = random_direction(init_vertices.device)
    else:
        rd = -rd
    
    # Execute pushing motion
    current_frame = 0
    target_changes = torch.zeros((n_frames, n_ctrl_parts, 3), dtype=torch.float32, device=robot_controller.device)
    
    while current_frame < n_frames:
        move_duration = torch.randint(30, 50, (1,)).item()
        move_duration = min(move_duration, n_frames - current_frame)

        speed = torch.rand(1).item() * (min_speed) + min_speed
        
        target_changes[current_frame:current_frame + move_duration, 0] = speed * rd
        
        current_frame += move_duration
        
        pause_duration = torch.randint(0, 10, (1,)).item()
        current_frame += pause_duration

    return initial_translation, target_changes, initial_finger, finger_changes


def simple_push_once(init_vertices, robot_controller, n_ctrl_parts):
    """
    Select a point on the rope, offset it by offset_dist, and move the pusher towards it at a fixed speed
    
    Args:
        init_vertices: Initial vertex positions
        robot_controller: Robot controller instance
        n_ctrl_parts: Number of control parts
        
    Returns:
        tuple: (current_pose, target_changes, initial_finger, finger_changes)
            - current_pose: numpy array [4,4] for robot transformation matrix
            - target_changes: numpy array for target movement
            - initial_finger: initial finger position
            - finger_changes: finger movement changes
    """
    offset_dist = 0.08
    keep_off = 0.03
    n_frames = 61
    speed = 0.014

    # finger is always closed
    initial_finger = 0.0
    finger_changes = torch.zeros(n_frames, dtype=torch.float32, device=robot_controller.device)
    
    # Get object points from the simulator
    max_attempts = 100
    for _ in range(max_attempts):
        # 1. Select a random point on the object
        random_idx = torch.randint(0, len(init_vertices), (1,)).item()
        selected_point = init_vertices[random_idx]
        
        # 2. Select a random direction
        rd = random_direction(init_vertices.device)
        
        # 3. Set starting position with offset
        start_position = selected_point + rd * offset_dist
        
        distances = torch.norm(init_vertices - start_position, dim=1)
        min_distance = torch.min(distances).item()
        
        if min_distance >= keep_off:
            # Valid position found
            break
    else:
        # If we couldn't find a valid position after max attempts
        raise RuntimeError("Could not find valid robot position after maximum attempts")
    
    # Calculate the initial translation to send the robot center to the selected point
    current_robot_center = robot_controller.get_current_center()
    initial_translation = start_position - current_robot_center

    rd = -rd

    # Execute pushing motion
    target_changes = torch.ones((n_frames, n_ctrl_parts, 3), dtype=torch.float32, device=robot_controller.device) * speed * rd

    return initial_translation, target_changes, initial_finger, finger_changes


def simple_lift_rope(init_vertices, robot_controller, n_ctrl_parts):
    """
    Lift the rope at a fixed speed
    
    Args:
        init_vertices: Initial vertex positions
        robot_controller: Robot controller instance
        n_ctrl_parts: Number of control parts
    """
    n_frames = 61 
    speed = 0.014
    wait_frames = 11

    # Pick a random point on the rope
    random_idx = torch.randint(0, len(init_vertices), (1,)).item()
    selected_point = init_vertices[random_idx]

    # Calculate the initial translation to send the robot center to the selected point
    current_robot_center = robot_controller.get_current_center()
    initial_translation = selected_point - current_robot_center

    # finger is fully open initially but closing
    initial_finger = 1.0 
    finger_changes = torch.ones(n_frames, dtype=torch.float32, device=robot_controller.device) * -0.05

    # Generate random movement sequence
    target_changes = torch.zeros((n_frames, n_ctrl_parts, 3), dtype=torch.float32, device=robot_controller.device)
        
    target_changes[wait_frames:n_frames, 0, 2] = -speed
    
    return initial_translation, target_changes, initial_finger, finger_changes


def lift_rope(init_vertices, robot_controller, n_ctrl_parts):
    """
    Grab the rope and go through a random sequence of ups, downs, and pauses
    Miss the rope 10% of the time (if the robot moves up before waiting for the fingers to close)
    Make sure the total displacement is never negative

    Args:
        init_vertices: Initial vertex positions
        robot_controller: Robot controller instance
        n_ctrl_parts: Number of control parts

    Return: 
        initial_translation: torch tensor [3] for translating the robot to the starting position
        target_changes: torch tensor [n_frames, n_ctrl_parts, 3] for robot movement (< 0 for up, > 0 for down)
        initial_finger: float for initial finger position (1.0 for open, 0.0 for closed)
        finger_changes: torch tensor [n_frames] for finger movement (-0.05 for closing, 0.05 for opening)
    """
    n_frames = 61 
    min_speed = 0.004

    # Pick a random point on the rope
    random_idx = torch.randint(0, len(init_vertices), (1,)).item()
    selected_point = init_vertices[random_idx]

    # Calculate the initial translation to send the robot center to the selected point
    current_robot_center = robot_controller.get_current_center()
    initial_translation = selected_point - current_robot_center

    # finger is fully open initially but closing
    initial_finger = 1.0 
    finger_changes = torch.ones(n_frames, dtype=torch.float32, device=robot_controller.device) * -0.05

    # Generate random movement sequence
    target_changes = torch.zeros((n_frames, n_ctrl_parts, 3), dtype=torch.float32, device=robot_controller.device)
    
    # Wait 9-11 frames at start to let fingers close
    wait_frames = torch.randint(9, 12, (1,)).item()
    
    # 10% chance to miss by moving up too early
    miss = torch.rand(1).item() < 0.1
    if miss:
        wait_frames = torch.randint(0, 2, (1,)).item()
        
    # After waiting, alternate between up movements and pauses
    current_frame = wait_frames
    total_displacement = 0.0
    
    while current_frame < n_frames:
        # Random up movement duration
        move_duration = torch.randint(5, 50, (1,)).item()
        move_duration = min(move_duration, n_frames - current_frame)

        if total_displacement > 0.008 * move_duration: 
            move_down = torch.rand(1).item() < 0.5
            if move_down:
                speed = torch.rand(1).item() * (min_speed) - 2 * min_speed
            else:
                speed = torch.rand(1).item() * (min_speed) + min_speed
        else: # can't move below the table
            speed = torch.rand(1).item() * (min_speed) + min_speed
        
        # Apply movement to first control part (index 0)
        target_changes[current_frame:current_frame + move_duration, 0, 2] = -speed
        total_displacement += speed * move_duration
        
        current_frame += move_duration
        
        pause_duration = torch.randint(0, 10, (1,)).item()
        current_frame += pause_duration

    return initial_translation, target_changes, initial_finger, finger_changes


def lift_air(init_vertices, robot_controller, n_ctrl_parts):
    """
    Similar to lift_rope but positions robot to miss the rope.
    Places robot near but not between fingers, ensuring no collision with rope.

    Args:
        init_vertices: Initial vertex positions
        robot_controller: Robot controller instance
        n_ctrl_parts: Number of control parts

    Return:
        initial_translation: torch tensor [3] for translating robot to miss position
        target_changes: torch tensor [n_frames, n_ctrl_parts, 3] for robot movement
        initial_finger: float for initial finger position (1.0 for open, 0.0 for closed)
        finger_changes: torch tensor [n_frames] for finger movement
    """
    n_frames = 61
    min_speed = 0.004
    offset_dist = 0.1  # 10cm offset
    keep_off = 0.07    # Center must be at least 7cm from any rope point

    # Find valid starting position away from rope
    max_attempts = 100
    for _ in range(max_attempts):
        # Select random point on rope
        random_idx = torch.randint(0, len(init_vertices), (1,)).item()
        selected_point = init_vertices[random_idx]
        
        # Select random direction for offset
        rd = random_direction(init_vertices.device)
        
        # Set starting position with offset
        start_position = selected_point + rd * offset_dist
        
        # Check distance to all rope points
        distances = torch.norm(init_vertices - start_position, dim=1)
        min_distance = torch.min(distances).item()
        
        if min_distance >= keep_off:
            # Valid position found
            break
    else:
        raise RuntimeError("Could not find valid robot position after maximum attempts")

    # Calculate initial translation
    current_robot_center = robot_controller.get_current_center()
    initial_translation = start_position - current_robot_center

    # Finger motion same as lift_rope
    initial_finger = 1.0
    finger_changes = torch.ones(n_frames, dtype=torch.float32, device=robot_controller.device) * -0.05

    # Generate movement sequence
    target_changes = torch.zeros((n_frames, n_ctrl_parts, 3), dtype=torch.float32, device=robot_controller.device)
    
    # Short wait at start
    wait_frames = torch.randint(0, 5, (1,)).item()
    
    # Execute lifting motion
    current_frame = wait_frames
    
    while current_frame < n_frames:
        move_duration = torch.randint(5, 16, (1,)).item()
        move_duration = min(move_duration, n_frames - current_frame)
        
        move_down = torch.rand(1).item() < 0.5 # we don't need strict checks since the robot isn't grasping the rope
        if move_down:
            speed = torch.rand(1).item() * (min_speed) - 2 * min_speed
        else:
            speed = torch.rand(1).item() * (min_speed) + min_speed
        
        target_changes[current_frame:current_frame + move_duration, 0, 2] = -speed
        
        current_frame += move_duration
        
        pause_duration = torch.randint(0, 5, (1,)).item()
        current_frame += pause_duration

    return initial_translation, target_changes, initial_finger, finger_changes


# Dictionary mapping function names to functions for easy lookup
ACTION_FUNCTIONS = {
    'push_once': push_once,
    'simple_push_once': simple_push_once,
    'simple_lift_rope': simple_lift_rope,
    'lift_rope': lift_rope,
    'lift_air': lift_air,
} 