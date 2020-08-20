
import setup_path
import airsim
import logging
import numpy as np
import time
import os
import pprint
import tempfile
import math
from math import *
# from scipy.misc import imsave
import imageio
# import cv2

from abc import ABC, abstractmethod
from path_planner import *
from squaternion import Quaternion


class AbstractClassGetNextVec(ABC):
    @abstractmethod
    def get_next_vec(self, depth, obj_sz, goal, pos):
        print("Some implementation!")
        yaw = 0
        return pos, yaw


class ReactiveController(AbstractClassGetNextVec):
    def get_next_vec(self, depth, obj_sz, goal, pos):
        print("Some implementation!")
        return


class AvoidLeft(AbstractClassGetNextVec):

    def __init__(self, hfov=radians(90), coll_thres=5, yaw=0, limit_yaw=5, step=0.1):
        self.hfov = hfov
        self.coll_thres = coll_thres
        self.yaw = yaw
        self.limit_yaw = limit_yaw
        self.step = step

    def get_next_vec(self, depth, obj_sz, goal, pos, img2d):
        [h, w] = np.shape(depth)
        [roi_h, roi_w] = compute_bb((h, w), obj_sz, self.hfov, self.coll_thres)

        # compute vector, distance and angle to goal
        t_vec, t_dist, t_angle = get_vec_dist_angle(goal, pos[:-1])

        # compute box of interest
        img2d_box = img2d[int((h - roi_h) / 2):int((h + roi_h) / 2), int((w - roi_w) / 2):int((w + roi_w) / 2)]

        # scale by weight matrix (optional)
        # img2d_box = np.multiply(img2d_box,w_mtx)

        # detect collision
        if np.min(img2d_box) < self.coll_thres:
            self.yaw = self.yaw - radians(self.limit_yaw)
        else:
            self.yaw = self.yaw + min(t_angle - self.yaw, radians(self.limit_yaw))

        pos[0] = pos[0] + self.step * cos(self.yaw)
        pos[1] = pos[1] + self.step * sin(self.yaw)

        return pos, self.yaw, t_dist


# compute resultant normalized vector, distance and angle
def get_vec_dist_angle(goal, pos):
    vec = np.array(goal) - np.array(pos)
    dist = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    angle = math.atan2(vec[1], vec[0])
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return vec / dist, dist, angle


# def get_local_goal(v, pos, theta):
#     return goal


# compute bounding box size
def compute_bb(image_sz, obj_sz, hfov, distance):
    vfov = hfov2vfov(hfov, image_sz)
    box_h = ceil(obj_sz[0] * image_sz[0] / (math.tan(hfov / 2) * distance * 2))
    box_w = ceil(obj_sz[1] * image_sz[1] / (math.tan(vfov / 2) * distance * 2))
    return box_h, box_w


# convert horizonal fov to vertical fov
def hfov2vfov(hfov, image_sz):
    aspect = image_sz[0] / image_sz[1]
    vfov = 2 * math.atan(tan(hfov / 2) * aspect)
    return vfov


# matrix with all ones
def equal_weight_mtx(roi_h, roi_w):
    return np.ones((roi_h, roi_w))


# matrix with max weight in center and decreasing linearly with distance from center
def linear_weight_mtx(roi_h, roi_w):
    w_mtx = np.ones((roi_h, roi_w))
    for j in range(0, roi_w):
        for i in range(j, roi_h - j):
            w_mtx[j:roi_h - j, i:roi_w - i] = (j + 1)
    return w_mtx


# matrix with max weight in center and decreasing quadratically with distance from center
def square_weight_mtx(roi_h, roi_w):
    w_mtx = np.ones((roi_h, roi_w))
    for j in range(0, roi_w):
        for i in range(j, roi_h - j):
            w_mtx[j:roi_h - j, i:roi_w - i] = (j + 1) * (j + 1)
    return w_mtx


def print_stats(img):
    print('Avg: ', np.average(img))
    print('Min: ', np.min(img))
    print('Max: ', np.max(img))
    print('Img Sz: ', np.size(img))


def generate_depth_viz(img, thres=0):
    if thres > 0:
        img[img > thres] = thres
    else:
        img = np.reciprocal(img)
    return img


def generate_contrast_viz(img, surface=0):
    if surface > 0:
        img[img < surface] = 0
        img[img >= surface] = 254
    else:
        img = np.reciprocal(img)
    return img



def plot_depth_cam(img, max_depth=20):
    img_plot = generate_depth_viz(img, max_depth)
    img_plot = (img_plot / max_depth) * 254
    img_plot = img_plot.astype(np.uint8)
    cv2.imshow('image', img_plot)
    cv2.waitKey(1)


def get_topview_image(hight=100, start_pos=[0, 0], drone=None, tmp_dir="airsim_drone"):
    drone.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(start_pos[0], start_pos[1], -hight),
                                         airsim.to_quaternion(0, 0, 0)), True, vehicle_name='Drone1')
    image_responses = drone.client.simGetImages([
        airsim.ImageRequest("bottom_center", airsim.ImageType.DepthPlanner, True)], vehicle_name="Drone1")

    image_response = image_responses[0]
    img1d = image_response.image_data_float
    img2d = np.reshape(img1d, (image_response.height, image_response.width))

    filename = os.path.join(tmp_dir, "depth_" + str(hight) + '.png')
    imageio.imwrite(os.path.normpath(os.path.join(tmp_dir, "depth_" + str(hight) + '.png')),
                    generate_contrast_viz(img2d, hight))

    # plt.imshow(img2d)
    # plt.show()

    real_image_responses = drone.client.simGetImages([
        airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)], vehicle_name="Drone1")
    img1d = np.fromstring(real_image_responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(real_image_responses[0].height, real_image_responses[0].width, 3)

    # plt.imshow(img_rgb)
    # plt.show()

    # airsim.write_png(os.path.normpath(os.path.join(tmp_dir, "real_" + str(hight) + '.png')), img_rgb)
    imageio.imwrite(os.path.normpath(os.path.join(tmp_dir, "real_" + str(hight) + '.png')), img_rgb)

    return img_rgb, filename


def find_corners(filename, proportion_x=1, proportion_y=1, tmp_dir="airsim_drone"):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.imread('airsim_drone/real_190.png')

    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []
    for contour in contours:
        min_x = gray.shape[0];
        max_x = 0;
        min_y = gray.shape[1];
        max_y = 0
        for contou in contour[1:]:
            img[contou[0][1], contou[0][0]] = [0, 0, 255]
            min_x = contou[0][0] if min_x > contou[0][0] else min_x
            max_x = contou[0][0] if max_x < contou[0][0] else max_x
            min_y = contou[0][1] if min_y > contou[0][1] else min_y
            max_y = contou[0][1] if max_y < contou[0][1] else max_y

        # Centering the obstacle
        O = np.array([(min_x - gray.shape[1] / 2) * proportion_x, (min_y - gray.shape[0] / 2) * proportion_y,
                      (max_x - gray.shape[1] / 2) * proportion_x, (max_y - gray.shape[0] / 2) * proportion_y])
        obstacles.append(O)
    cv2.imwrite(os.path.normpath(os.path.join(tmp_dir, "corners" + '.png')), img)
    return obstacles


def calc_relative_heading(drone_euler, car_euler):
    yaw_drone = drone_euler[2] if drone_euler[2] < 180 else drone_euler[2] - 180
    yaw_car = car_euler[0] if car_euler[0] < 180 else car_euler[0] - 180

    return abs(yaw_car - yaw_drone)


def get_state_vector(drone, car, planner=None):
    drone = drone.client.getMultirotorState(drone.name)
    car = car.client.getCarState(car.name)
    state_dict = {}
    drone_pose = drone.kinematics_estimated.position.to_numpy_array()
    car_pose = car.kinematics_estimated.position.to_numpy_array()
    drone_vel = drone.kinematics_estimated.linear_velocity.to_numpy_array()
    car_vel = car.kinematics_estimated.linear_velocity.to_numpy_array()

    drone_orientation = Quaternion(*drone.kinematics_estimated.orientation.to_numpy_array())
    car_orientation = Quaternion(*car.kinematics_estimated.orientation.to_numpy_array())

    relative_dist = drone.kinematics_estimated.position.distance_to(car.kinematics_estimated.position)
    relative_vel = drone.kinematics_estimated.linear_velocity.distance_to(car.kinematics_estimated.linear_velocity)

    relative_heading = calc_relative_heading(drone_euler=drone_orientation.to_euler(degrees=True),
                                             car_euler=car_orientation.to_euler(degrees=True))

    drone_speed = np.linalg.norm(drone_vel)
    try:
        los = 1 if planner.X.collision_free(drone_pose[:2], car_pose[:2], planner.r) else 0
    except Exception:
        print('planner not defined yet')
        los = -1

    state_vec = [relative_dist, relative_vel, relative_heading, drone_speed, los]
    state_dict['drone'] = {'pose': drone_pose, 'velocity': drone_vel, 'orientation': drone_orientation}
    state_dict['car'] = {'pose': car_pose, 'velocity': car_vel, 'orientation': car_orientation}
    return state_vec, state_dict


def setup_logger(name, dir_name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s')
    if not os.path.exists(f'./{dir_name}/'):
        os.makedirs(f'./{dir_name}/')

    log_file = './' + dir_name + '/' + log_file

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

