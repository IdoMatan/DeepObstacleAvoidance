import setup_path
import airsim

import numpy as np
import atexit
from utils import plot_depth_cam, bounding_box, prepare_for_yolo
import os
import imageio
import time

class Drone:
    def __init__(self, name='Drone1', mode=None, uav_size=(0.29*3, 0.98*2)):
        self.client = airsim.MultirotorClient(port=41451)
        self.name = name
        self.init_client()
        self.current_goal = [0, 0, 0]
        self.current_pose = None
        atexit.register(self.disconnect)
        self.mode = mode
        self.predictControl = None
        self.uav_size = uav_size
        self.leader = None
        self.poses = []
        self.yaws = []

    def init_client(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)

    def move(self, pos, yaw):
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[1], pos[0], pos[2]), airsim.to_quaternion(0, 0, yaw)),
                                      True, vehicle_name=self.name)

    def set_speed(self, speed):
        '''
        send a command to drone to keep flying to current goal but with new speed
        :arg speed: scalar value between 0-1 (0 is hover in place)
        '''
        height_default = -2
        speed_const = 10
        # self.client.enableApiControl(True, self.name)
        self.client.moveToPositionAsync(*self.current_goal, height_default, speed*speed_const, vehicle_name=self.name)

    def dist(self, position):
        if self.current_pose is None:
            return False
        else:
            return np.linalg.norm(self.current_pose[:2] - position)

    def get_img2d(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.DepthPlanner, True)])
        response = responses[0]
        # get numpy array
        img1d = response.image_data_float
        # reshape array to 2D array H X W
        try:
            img2d = np.reshape(img1d, (response.height, response.width))
            # print(response.camera_position)
            return img2d
        except:
            return False

    def get_img2d_scene(self):
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        try:
            img2d = img1d.reshape(response.height, response.width, 3)
            return img2d
        except:
            None
        # img2d = np.flipud(img2d)

    def get_img2d_segmentation(self):
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)],
                                             vehicle_name=self.name)
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        try:
            img2d = img1d.reshape(response.height, response.width, 3)
            import cv2
            cv2.imshow('image', img2d)
            cv2.waitKey(1)
            return img2d
        except:
            None


    def disconnect(self):
        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)

    def step(self, goal, pos):
        img2d = self.get_img2d()
        # plot_depth_cam(img2d)
        [pos, yaw, target_dist] = self.predictControl.get_next_vec(img2d, self.uav_size, goal, pos)
        self.move(pos, yaw)
        return pos, yaw, target_dist

    def follow(self, pos, yaw):
        self.poses.append(pos)
        self.yaws.append(yaw)
        self.move(self.poses.pop(0), self.yaws.pop(0))

    def save_leading_pic(self):
        filename = str(int(np.floor(time.time())))
        depth_img2d = self.get_img2d()
        img2d = self.get_img2d_scene()
        # self.get_img2d_segmentation()
        if img2d is not None:
            x_min, x_max, y_min, y_max = bounding_box(depth_img2d)
            if 2 < (x_max - x_min) < img2d.shape[1] and 2 < (y_max - y_min) < img2d.shape[0]:
                prepare_for_yolo(x_min, x_max, y_min, y_max, filename='./yolo_data/' + filename + '.txt')
                imageio.imwrite(os.path.normpath(os.path.join('./yolo_data/' + filename + '.jpg')), img2d)

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1)
                ax.imshow(img2d)
                # x_min, x_max, y_min, y_max = bounding_box(self.get_img2d())
                import matplotlib.patches as patches
                rect = patches.Rectangle((y_min, x_min), y_max-y_min, x_max-x_min, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                plt.show()

class Car:
    def __init__(self, name='Car1'):
        self.client = airsim.CarClient(port=41452)
        self.name = name
        self.init_client()
        self.car_controls = airsim.CarControls()
        self.current_pose = None
        atexit.register(self.disconnect)

    def init_client(self):
        self.client.confirmConnection()
        # self.client.enableApiControl(True, self.name)

    def move(self, pos, yaw):
        # pos Z coordinate is overriden to -1 (or 0, need to test)
        self.client.enableApiControl(True, self.name)
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0], pos[1], 0), airsim.to_quaternion(0, 0, yaw)),
                                      True, vehicle_name=self.name)
        self.client.enableApiControl(False, self.name)

    def dist(self, position):
        if self.current_pose is None:
            return False
        else:
            return np.norm(self.current_pose - position)

    def disconnect(self):
        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)
