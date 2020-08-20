import setup_path
import airsim

import numpy as np
import atexit


class Drone:
    def __init__(self, name='Drone1'):
        self.client = airsim.MultirotorClient(port=41451)
        self.name = name
        self.init_client()
        self.current_goal = [0, 0, 0]
        self.current_pose = None
        atexit.register(self.disconnect)

    def init_client(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)

    def move(self, pos, yaw):
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0], pos[1], pos[2]), airsim.to_quaternion(0, 0, yaw)),
                                      True, vehicle_name=self.name)

    def set_speed(self, speed):
        '''
        send a command to drone to keep flying to current goal but with new speed
        :arg speed: scalar value between 0-1 (0 is hover in place)
        '''
        height_default = -2
        speed_const = 10
        self.client.enableApiControl(True, self.name)
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

    def disconnect(self):
        self.client.armDisarm(False, self.name)
        self.client.enableApiControl(False, self.name)


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
