import os
import sys
import configparser
import cv2


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from utility.arm_utils import *
from utility.utils import *
from utility.gelsight import *

# === Read config file ===
config_path = os.path.join(os.path.dirname(__file__), 'robot.conf')
parser = configparser.ConfigParser()
parser.read(config_path)

# === Robot Setup ===
try:
    ip = parser.get('xArm', 'ip')
    speed = parser.getfloat('xArm', 'speed')
    data_collect = parser.getfloat('xArm', 'data_collect')
    up = parser.getfloat('xArm', 'up')
    down = parser.getfloat('xArm', 'down')

except Exception as e:
    print("Failed to read 'ip' or 'speed' from robot.conf:", e)
    sys.exit(1)

#Read from Gelsight
#stream = init_gelsight(src=0)

# === Connect to robot ===
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)

# change data collect to 1 in robot.conf for data collection
# if data_collect == 1:
#     setup_data_folder()

# Get initial Pose
origin = get_tcp_pose(arm)
print("Current TCP pose:", origin)


poses = circular_path_gen(10.0)
poses = palpation_path(poses, up, down)
move_along_path(arm, origin, poses, speed)


# grabbed, frame = stream.read()
# if grabbed:
#     cv2.imshow("Gelsight Frame", frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()






# === Disconnect ===
move_to_relative_pose(arm, origin, [0,0,0,0,0,0], speed)
arm.disconnect()
print("Motion complete. Disconnected.")
