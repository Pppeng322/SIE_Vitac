import math
from xarm.wrapper import XArmAPI

def get_tcp_pose(arm: XArmAPI):
    '''
    Get current TCP pose
    '''
    return arm.position


def move_to_relative_pose(arm: XArmAPI, original_pose: list, relative_move: list, speed: float):
    '''
    #Move to a pose relative to the original pose
    '''
    if len(original_pose) != 6 or len(relative_move) != 6:
        raise ValueError("Both original_pose and relative_move must be lists of 6 elements")

    new_pose = [o + d for o, d in zip(original_pose, relative_move)]

    arm.set_position(x=new_pose[0], y=new_pose[1], z=new_pose[2], speed=speed, wait=True)
    print('Moved to:',new_pose)


def move_along_path(arm, init_pose: list, relative_moves: list, speed: float):
    """
    Move the robot along a list of poses relative to init_pose (absolute motion).

    Parameters:
        arm: XArmAPI instance
        init_pose: list of 6 floats [x, y, z, rx, ry, rz]
        relative_moves: list of 6-element lists [dx, dy, dz, drx, dry, drz]
        speed: float, motion speed in mm/s
    """
    if len(init_pose) != 6:
        raise ValueError("init_pose must be a list of 6 elements")

    for idx, rel in enumerate(relative_moves):
        if len(rel) != 6:
            raise ValueError(f"relative_moves[{idx}] must have 6 elements")

        # Calculate new absolute pose from initial pose + relative movement
        target_pose = [i + r for i, r in zip(init_pose, rel)]

        print(f"Moving to pose  {target_pose}")
        arm.set_position(
            x=target_pose[0],
            y=target_pose[1],
            z=target_pose[2],
            roll=target_pose[3],
            pitch=target_pose[4],
            yaw=target_pose[5],
            speed=speed,
            wait=True
        )

