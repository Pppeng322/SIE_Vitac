import os
import math
from datetime import datetime


def setup_data_folder(base_path: str = None) -> str:
    """
    Creates a 'data' directory (if it doesn't exist), a timestamped subdirectory, and
    'pose' and 'image' subfolders inside the timestamped directory.

    The timestamped subdirectory is named using the current time in YYYYMMDDHHMMSS format.

    Parameters:
        base_path (str): Optional base path under which to create 'data'.
                         Defaults to the current working directory.

    Returns:
        str: The timestamp string of the newly created subdirectory (YYYYMMDDHHMMSS).
    """
    # Determine base path
    if base_path is None:
        base_path = os.getcwd()

    # Ensure base_path exists
    os.makedirs(base_path, exist_ok=True)

    # Create 'data' directory
    data_dir = os.path.join(base_path, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    timestamp_dir = os.path.join(data_dir, timestamp)

    # Create timestamped directory and subfolders
    pose_dir = os.path.join(timestamp_dir, 'pose')
    image_dir = os.path.join(timestamp_dir, 'image')
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    return timestamp


def pixel_to_tcp(
    pixel: tuple,
    center_pixel: tuple,
    mm_per_pixel: float,
    tcp_xy: tuple = (0.0, 0.0)
) -> tuple:
    """
    Convert a camera pixel coordinate to a robot end-effector (EE) XY position.

    Parameters:
        pixel: (u, v) pixel coordinates in the image.
        center_pixel: (u0, v0) pixel coordinates that correspond to the EE origin.
        mm_per_pixel: scale factor, how many millimeters each pixel represents.
        ee_origin_xy: (x0, y0) the EE's XY in robot frame that maps to center_pixel;
                      defaults to (0,0).

    Returns:
        (x, y): the EE XY position in millimeters.
    """
    u, v = pixel
    u0, v0 = center_pixel
    x0, y0 = tcp_xy

    # Pixel offset from center
    du = u - u0
    dv = v - v0

    # Convert to mm (you may need to flip sign on dv depending on your camera orientation)
    dx = du * mm_per_pixel
    dy = dv * mm_per_pixel

    # Absolute EE position
    x = x0 + dx
    y = y0 + dy

    return x, y

def circular_path_gen(radius: float, num_points: int = 5):
    """
    Generate a list of relative poses forming a circular path on the XY plane.

    Parameters:
        radius: radius of the circle (in meters)
        num_points: number of waypoints to generate along the circle

    Returns:
        A list of relative poses: [[dx, dy, dz, drx, dry, drz], ...]
    """
    if radius <= 0 or num_points < 3:
        raise ValueError("Radius must be positive and num_points >= 3")

    poses = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        poses.append([x, y, 0, 0, 0, 0])  # XY plane, no rotation

    return poses

def palpation_path(path: list, above: float, below: float) -> list:
    """
    Given a list of poses [x, y, z, rx, ry, rz], generate a new path that,
    for each original pose:
      1. Starts at `above` mm above the pose
      2. Moves down to `below` mm below the pose
      3. Moves back up to `above` mm above the pose

    The result concatenates these 3 steps for each waypoint in the input path.

    Parameters:
        path: List of 6‐element poses [x, y, z, rx, ry, rz].
        above: Height above each pose (mm).
        below: Depth below each pose (mm).

    Returns:
        List of 6‐element poses forming the zig–zag trajectory.
    """
    new_path = []
    for pose in path:
        if len(pose) != 6:
            raise ValueError("Each pose must be a 6‐element list")
        x, y, z, rx, ry, rz = pose
        # 1) above
        new_path.append([x, y, z + above, rx, ry, rz])
        # 2) below
        new_path.append([x, y, z - below, rx, ry, rz])
        # 3) back above
        new_path.append([x, y, z + above, rx, ry, rz])
    return new_path


def get_gelsight_frame(src=0):
    """
    Uses the Gelsight camera class to grab a single frame.
    Returns:
        frame (ndarray) or None if capture failed.
    """
    # Start the camera stream
    stream = Gelsight(src=src).start()
    # Read one frame
    grabbed, frame = stream.read()
    # Stop the stream
    stream.stop()
    if not grabbed:
        print(f"Error: failed to grab frame from camera {src}")
        return None
    return frame