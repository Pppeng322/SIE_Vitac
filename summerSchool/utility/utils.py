import os
import math
from datetime import datetime
import numpy as np

from gelsight import Gelsight


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


def relative_to_origin(point, origin):
    """
    point: [x, y, z, rx, ry, rz]
    origin: [x, y, z, rx, ry, rz]

    Returns: [x - origin_x, y - origin_y, z, rx, ry, rz]
    """
    rel_x = point[0] - origin[0]
    rel_y = point[1] - origin[1]
    rel_z = point[2] - origin[2]
    rel_rx = point[3] - origin[3]
    rel_ry = point[4] - origin[4]
    rel_rz = point[5] - origin[5]

    return [rel_x, rel_y, rel_z, rel_rx, rel_ry, rel_rz]


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

def generate_scan_path(corners, origin, rows, cols):
    """
    corners: list of 4 tuples (x,y,z,rx,ry,rz)
        top_left, top_right, bottom_right, bottom_left
    origin: tuple (x,y,z,rx,ry,rz) treated as (0,0)
    rows, cols: grid size
    returns: list of (x,y,z,rx,ry,rz) relative to origin
    """

    # Only use x, y
    origin_x, origin_y = origin[0], origin[1]

    # Extract only x,y for corners
    tl = np.array([corners[0][0], corners[0][1]], dtype=np.float32)
    tr = np.array([corners[1][0], corners[1][1]], dtype=np.float32)
    br = np.array([corners[2][0], corners[2][1]], dtype=np.float32)
    bl = np.array([corners[3][0], corners[3][1]], dtype=np.float32)

    coords_list = []

    for r in range(rows):
        v = r / (rows - 1) if rows > 1 else 0.0

        left_edge_point = tl * (1 - v) + bl * v
        right_edge_point = tr * (1 - v) + br * v

        row_points = []
        for c in range(cols):
            u = c / (cols - 1) if cols > 1 else 0.0
            p = left_edge_point * (1 - u) + right_edge_point * u

            # Shift relative to origin
            rel_x = p[0] - origin_x
            rel_y = p[1] - origin_y

            # Add in (x,y,z,rx,ry,rz)
            row_points.append( (rel_x, rel_y, 0, 0, 0, 0) )

        # Snaking
        if r % 2 == 0:
            coords_list.extend(row_points)
        else:
            coords_list.extend(reversed(row_points))

    return coords_list

def find_corners(corners):
    """
    corners: list of 4 [x, y, z, rx, ry, rz]
    returns: list of 4 [x, y, z, rx, ry, rz] in order:
             top left, top right, bottom right, bottom left
    """

    if len(corners) != 4:
        raise ValueError(f"Expected exactly 4 corners, but got {len(corners)}")

    # Extract x and y
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]

    xs_sorted = sorted(xs)
    ys_sorted = sorted(ys)

    # Make sure there is range to define a rectangle
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        raise ValueError("Corners do not define a valid area (all x or all y identical)")

    left_x  = max(xs_sorted[0], xs_sorted[1])
    right_x = min(xs_sorted[2], xs_sorted[3])

    top_y    = max(ys_sorted[0], ys_sorted[1])
    bottom_y = min(ys_sorted[2], ys_sorted[3])

    # Use z, rx, ry, rz from the first point
    z, rx, ry, rz = corners[0][2], corners[0][3], corners[0][4], corners[0][5]

    # Return rectangle corners in required order
    return [
        [left_x,  top_y,    z, rx, ry, rz],  # top left
        [right_x, top_y,    z, rx, ry, rz],  # top right
        [right_x, bottom_y, z, rx, ry, rz],  # bottom right
        [left_x,  bottom_y, z, rx, ry, rz]   # bottom left
    ]


import random


def generate_noisy_rectangle_corners(origin, width, height, noise_range):
    """
    origin: [x, y, z, rx, ry, rz]
    width: rectangle width along x
    height: rectangle height along y
    noise_range: maximum noise in mm (+/-)

    Returns list of 4 corners:
    [top left, top right, bottom right, bottom left]
    Each as [x, y, z, rx, ry, rz]
    """
    ox, oy, oz, orx, ory, orz = origin

    # Define ideal corners (before noise)
    ideal_corners = [
        [ox, oy],  # top left
        [ox + width, oy],  # top right
        [ox + width, oy + height],  # bottom right
        [ox, oy + height]  # bottom left
    ]

    # Add noise to x and y
    noisy_corners = []
    for x, y in ideal_corners:
        nx = x + random.uniform(-noise_range, noise_range)
        ny = y + random.uniform(-noise_range, noise_range)
        noisy_corners.append([nx, ny, oz, orx, ory, orz])

    return noisy_corners
