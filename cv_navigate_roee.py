
# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import setup_path 
import airsim

import numpy as np
import time
import os
import pprint
import tempfile
import math
from math import *
# from scipy.misc import imsave
import imageio
import cv2

from abc import ABC, abstractmethod
 
#define abstract class to return next vector in the format (x,y,yaw)
class AbstractClassGetNextVec(ABC):
    
    @abstractmethod
    def get_next_vec(self, depth, obj_sz, goal, pos):
        print("Some implementation!")
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

    def get_next_vec(self, depth, obj_sz, goal, pos):
        [h, w] = np.shape(depth)
        [roi_h, roi_w] = compute_bb((h, w), obj_sz, self.hfov, self.coll_thres)

        # compute vector, distance and angle to goal
        t_vec, t_dist, t_angle = get_vec_dist_angle(goal, pos[:-1])

        # compute box of interest
        img2d_box = img2d[int((h-roi_h)/2):int((h+roi_h)/2), int((w-roi_w)/2):int((w+roi_w)/2)]

        # scale by weight matrix (optional)
        # img2d_box = np.multiply(img2d_box,w_mtx)
    
        # detect collision
        if np.min(img2d_box) < coll_thres:
            self.yaw = self.yaw - radians(self.limit_yaw)
        else:
            self.yaw = self.yaw + min(t_angle-self.yaw, radians(self.limit_yaw))

        pos[0] = pos[0] + self.step*cos(self.yaw)
        pos[1] = pos[1] + self.step*sin(self.yaw)

        return pos, self.yaw, t_dist


class AvoidLeftIgonreGoal(AbstractClassGetNextVec):

    def __init__(self, hfov=radians(90), coll_thres=5, yaw=0, limit_yaw=5, step=0.1):
        self.hfov = hfov
        self.coll_thres = coll_thres
        self.yaw = yaw
        self.limit_yaw = limit_yaw
        self.step = step

    def get_next_vec(self, depth, obj_sz, goal, pos):
        [h, w] = np.shape(depth)
        [roi_h, roi_w] = compute_bb((h, w), obj_sz, self.hfov, self.coll_thres)

        # compute box of interest
        img2d_box = img2d[int((h-roi_h)/2):int((h+roi_h)/2),int((w-roi_w)/2):int((w+roi_w)/2)]

        # detect collision
        if np.min(img2d_box) < coll_thres:
            self.yaw = self.yaw - radians(self.limit_yaw)

        pos[0] = pos[0] + self.step*cos(self.yaw)
        pos[1] = pos[1] + self.step*sin(self.yaw)

        return pos, self.yaw, 100


class AvoidLeftRight(AbstractClassGetNextVec):
    def get_next_vec(self, depth, obj_sz, goal, pos):
        print("Some implementation!")
        # Same as above but decide to go left or right based on average or some metric like that
        return


# compute resultant normalized vector, distance and angle
def get_vec_dist_angle(goal, pos):
    vec = np.array(goal) - np.array(pos)
    dist = math.sqrt(vec[0]**2 + vec[1]**2)
    angle = math.atan2(vec[1],vec[0])
    if angle > math.pi:
        angle -= 2*math.pi
    elif angle < -math.pi:
        angle += 2*math.pi
    return vec/dist, dist, angle


def get_local_goal(v, pos, theta):
    return goal


# compute bounding box size
def compute_bb(image_sz, obj_sz, hfov, distance):
    vfov = hfov2vfov(hfov,image_sz)
    box_h = ceil(obj_sz[0] * image_sz[0] / (math.tan(hfov/2)*distance*2))
    box_w = ceil(obj_sz[1] * image_sz[1] / (math.tan(vfov/2)*distance*2))
    return box_h, box_w


# convert horizonal fov to vertical fov
def hfov2vfov(hfov, image_sz):
    aspect = image_sz[0]/image_sz[1]
    vfov = 2*math.atan(tan(hfov/2) * aspect)
    return vfov


# matrix with all ones
def equal_weight_mtx(roi_h, roi_w):
    return np.ones((roi_h, roi_w))


# matrix with max weight in center and decreasing linearly with distance from center
def linear_weight_mtx(roi_h, roi_w):
    w_mtx = np.ones((roi_h, roi_w))
    for j in range(0, roi_w):
        for i in range(j, roi_h-j):
            w_mtx[j:roi_h-j, i:roi_w-i] = (j+1)
    return w_mtx


# matrix with max weight in center and decreasing quadratically with distance from center
def square_weight_mtx(roi_h, roi_w):
    w_mtx = np.ones((roi_h, roi_w))
    for j in range(0, roi_w):
        for i in range(j, roi_h-j):
            w_mtx[j:roi_h-j, i:roi_w-i] = (j+1)*(j+1)
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


def moveUAV(client, pos, yaw):
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0], pos[1], pos[2]), airsim.to_quaternion(0, 0, yaw)), True) 


def plot_depth_cam(img, max_depth=20):
    img_plot = generate_depth_viz(img, max_depth)
    img_plot = (img_plot / max_depth) * 254
    img_plot = img_plot.astype(np.uint8)
    cv2.imshow('image', img_plot)
    cv2.waitKey(1)


def get_topview_image(hight=100):
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, -hight), airsim.to_quaternion(0, 0, 3.14)), True)
    image_responses = client.simGetImages([
        airsim.ImageRequest("bottom_center", airsim.ImageType.DepthPlanner, True)])
    image_response = image_responses[0]
    img1d = image_response.image_data_float
    img2d = np.reshape(img1d, (image_response.height, image_response.width))

    imageio.imwrite(os.path.normpath(os.path.join(tmp_dir, "depth_" + str(hight) + '.png')), generate_contrast_viz(img2d, hight))
    filename = os.path.join(tmp_dir, "depth_" + str(hight) + '.png')
    return img2d, filename


def find_corners(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []
    for contour in contours:
        min_x = 255; max_x = 0; min_y = 255; max_y = 0
        for contou in contour[1:]:
            img[contou[0][1], contou[0][0]] = [0, 0, 255]
            min_x = contou[0][0] if min_x > contou[0][0] else min_x
            max_x = contou[0][0] if max_x < contou[0][0] else max_x
            min_y = contou[0][1] if min_y > contou[0][1] else min_y
            max_y = contou[0][1] if max_y < contou[0][1] else max_y

        O = np.array([(min_x-128)*2, (min_y-72)*1.3, (max_x-128)*2, (max_y-72)*1.3])
        obstacles.append(O)
    cv2.imwrite(os.path.normpath(os.path.join(tmp_dir, "corners" + '.png')), img)
    return obstacles


def rrt_navigation(X_dimensions = np.array([(-500, 500), (-200, 200)]),  #  X_dimensions = np.array([(-255, 255), (-145, 145)]),  # dimensions of Search Space
                   x_init=(0, 0),                                  # starting location
                   x_goal=(100, 100),                              # goal location
                   obstacles=[],
                   Q = np.array([(8, 4)]),                         # length of tree edges
                   r=1,                                            # length of smallest edge to check for intersection with obstacles
                   max_samples=1024,                               # max number of samples to take before timing out
                   rewire_count=32,                                # optional, number of nearby branches to rewire
                   prc=0.1,                                        # probability of checking for a connection to goal):
                   plot=True,
                   ):
    from computer_vision.rrt_src.rrt.rrt_star import RRTStar
    from computer_vision.rrt_src.search_space.search_space import SearchSpace
    from computer_vision.rrt_src.utilities.plotting import Plot
    from computer_vision.rrt_src.utilities.obstacle_generation import obstacle_generator

    # create Search Space
    X = SearchSpace(X_dimensions)
    obstacles = obstacles[1:]
    obstacle_generator(obstacles, X)

    # create rrt_search
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star()

    if plot:
        plot = Plot("rrt_star_2d_with_random_obstacles")

        plot.plot_tree(X, rrt.trees)
        if path is not None:
            plot.plot_path(X, path)
        plot.plot_obstacles(X, obstacles)
        plot.plot_start(X, x_init)
        plot.plot_goal(X, x_goal)
        plot.draw(auto_open=True)
    return path


pp = pprint.PrettyPrinter(indent=4)

client = airsim.VehicleClient()
client.confirmConnection()

tmp_dir = "airsim_drone"
# print ("Saving images to %s" % tmp_dir)
# airsim.wait_key('Press any key to start')

# Define start position, goal and size of UAV
pos = [0, 5, -1]  # start position x,y,z
goals = [[120, 0]]  # , [0, 100], [100, 100], [0, 0]]  # x,y
uav_size = [0.29*3, 0.98*2]  # height:0.29 x width:0.98 - allow some tolerance

# Define parameters and thresholds
hfov = radians(90)
coll_thres = 5
yaw = 0
limit_yaw = 5
step = 0.1

responses = client.simGetImages([
    airsim.ImageRequest("1", airsim.ImageType.DepthPlanner, True)])
response = responses[0]

# initial position
moveUAV(client, pos, yaw)

# predictControl = AvoidLeftIgonreGoal(hfov, coll_thres, yaw, limit_yaw, step)
predictControl = AvoidLeft(hfov, coll_thres, yaw, limit_yaw, step)

topview, filename = get_topview_image(190)
obstacles = find_corners(filename)
goals = rrt_navigation(obstacles=obstacles)
# goles = generate_navigation_graph()
moveUAV(client, pos, yaw)

# for z in range(10000):  # do few times
for goal in goals:  # do few times
    for z in range(10000):
        # get response
        responses = client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.DepthPlanner, True)])
        response = responses[0]

        # get numpy array
        img1d = response.image_data_float

        # reshape array to 2D array H X W
        try:
            img2d = np.reshape(img1d, (response.height, response.width))
            # print(response.camera_position)
        except:
            continue

        plot_depth_cam(img2d)

        [pos, yaw, target_dist] = predictControl.get_next_vec(img2d, uav_size, goal, pos)
        moveUAV(client, pos, yaw)

        if target_dist < 1:
            print('Target reached.')
            print(response.camera_position)
            # airsim.wait_key('Press any key to continue')
            break

    # write to png 
    # imageio.imwrite(os.path.normpath(os.path.join(tmp_dir, "depth_" + str(z) + '.png')), generate_depth_viz(img2d,5))

    # time.sleep(5)

# currently reset() doesn't work in CV mode. Below is the workaround
client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)














#################### OLD CODE
#    timer = 0
#    time_obs = 50
#    bObstacle = False

#    if (bObstacle):
#        timer = timer + 1
#        if timer > time_obs:
#            bObstacle = False
#            timer = 0
#    else:
#        yaw = target_angle

#    print (target_angle,target_vec,target_dist,x,y,goal[0],goal[1])


#    if (np.average(img2d_box) < coll_thres):
#        img2d_box_l = img2d_box = img2d[int((h-roi_h)/2):int((h+roi_h)/2),int((w-roi_w)/2)-50:int((w+roi_w)/2)-50]
#        img2d_box_r = img2d_box = img2d[int((h-roi_h)/2):int((h+roi_h)/2),int((w-roi_w)/2)+50:int((w+roi_w)/2)+50]
#        img2d_box_l_avg = np.average(np.multiply(img2d_box_l,w_mtx))
#        img2d_box_r_avg = np.average(np.multiply(img2d_box_r,w_mtx))
#        print('left: ', img2d_box_l_avg)
#        print('right: ', img2d_box_r_avg)
#        if img2d_box_l_avg > img2d_box_r_avg:
#            ##Go LEFT
#            #y_offset = y_offset-1
#            yaw = yaw - radians(10)
#            bObstacle = True
#        else:
#            ##Go RIGHT
#            #y_offset = y_offset+1
#            yaw = yaw + radians(10)
#            bObstacle = true
#        print('yaw: ', yaw)
