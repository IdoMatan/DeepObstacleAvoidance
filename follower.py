import airsim
import numpy as np
from math import *
import cv2
import os
import matplotlib.pyplot as plt


def moveUAV(client, pos, yaw):
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0]+6, pos[1], pos[2]+0.1), airsim.to_quaternion(0, 0, yaw)), True, vehicle_name="Drone1")
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0]-5, pos[1], pos[2]), airsim.to_quaternion(0, 0, yaw)), True, vehicle_name="Drone2")


def remove_background(client, dist=5, segmentation=True):
    # responses = client.simGetImages([
    #     airsim.ImageRequest("1", airsim.ImageType.DepthPlanner, True)])
    # responses = client.simGetImages([
    #     airsim.ImageRequest("1", airsim.ImageType.DepthPlanner, True)], vehicle_name="Drone2")
    if segmentation:
        responses = client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.Segmentation, False, False)], vehicle_name="Drone2")
        response = responses[0]
        # img1d = response.image_data_uint8
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        try:
            img2d = np.reshape(img1d, (response.height, response.width, 3))
            cv2.imwrite('Segmentation' + ".png", img2d)
        except:
            print('Image reshape error')
    # else:
        responses = client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.DepthPlanner, True)], vehicle_name="Drone2")
        response = responses[0]
        img1d = response.image_data_float
        try:
            img2d = np.reshape(img1d, (response.height, response.width))
            cv2.imwrite('depth' + ".png", img2d)
        except:
            print('Image reshape error')

    image = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)], vehicle_name="Drone2")
    image_rgb = image[0]
    img_rgb_1d = np.frombuffer(image_rgb.image_data_uint8, dtype=np.uint8)
    img_rgb = img_rgb_1d.reshape(image_rgb.height, image_rgb.width, 3)
    cv2.imwrite('original' + ".png", img_rgb)
    img2d[img2d < dist] = 0
    depth_img = np.repeat(img2d.reshape(image_rgb.height, image_rgb.width, 1), repeats=3, axis=2)
    test_img = np.copy(img_rgb)
    test_img[depth_img > 0] = 0
    cv2.imwrite('filtered' + ".png", test_img)


def find_corners(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    drone = np.where(gray > 0)

    max_x = drone[0].max()
    min_x = drone[0].min()
    max_y = drone[1].max()
    min_y = drone[1].min()

    img[min_x, min_y] = [0, 0, 255]
    img[max_x, max_y] = [0, 0, 255]

    cv2.imwrite(os.path.normpath(os.path.join("corners" + '.png')), img)


def main():
    client = airsim.VehicleClient()
    client.confirmConnection()

    client.enableApiControl(True, "Drone1")
    client.enableApiControl(True, "Drone2")
    client.armDisarm(True, "Drone1")
    client.armDisarm(True, "Drone2")

    tmp_dir = "airsim_drone"

    # Define start position, goal and size of UAV
    pos = [0, 5, -1]  # start position x,y,z
    goals = [[120, 0]]  # , [0, 100], [100, 100], [0, 0]]  # x,y
    uav_size = [0.29*3, 0.98*2]  # height:0.29 x width:0.98 - allow some tolerance

    # initial position
    moveUAV(client, pos, yaw=0)

    remove_background(client=client)
    find_corners('filtered.png')


if __name__ == '__main__':
    main()
