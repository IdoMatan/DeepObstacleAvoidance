# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

from itertools import tee

import numpy as np


def dist_between_points(a, b):
    """
    Return the Euclidean distance between two points
    :param a: first point
    :param b: second point
    :return: Euclidean distance between a and b
    """
    distance = np.linalg.norm(np.array(b) - np.array(a))
    return distance


def pairwise(iterable):
    """
    Pairwise iteration over iterable
    :param iterable: iterable
    :return: s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def es_points_along_line(start, end, r):
    """
    Equally-spaced points along a line defined by start, end, with resolution r
    :param start: starting point
    :param end: ending point
    :param r: maximum distance between points
    :return: yields points along line from start to end, separated by distance r
    """
    d = dist_between_points(start, end)
    n_points = int(np.ceil(d / r))
    if n_points > 1:
        step = d / (n_points - 1)
        for i in range(n_points):
            next_point = steer(start, end, i * step)
            yield next_point


def steer(start, goal, d):
    """
    Return a point in the direction of the goal, that is distance away from start
    :param start: start location
    :param goal: goal location
    :param d: distance away from start
    :return: point in the direction of the goal, distance away from start
    """
    start, end = np.array(start), np.array(goal)
    v = end - start
    u = v / (np.sqrt(np.sum(v ** 2)))
    steered_point = start + u * d
    return tuple(steered_point)


def get_dist(p1, p2):
    """
    Return the distance between 2 numpy formant points
    :param p1:
    :param p2:
    :return:
    """
    try:
        return np.linalg.norm(p2-p1)

    except Exception:
        return np.linalg.norm(np.array(p2) - np.array(p1))


def calc_angle_cost(P1,P2,P3):
    if sum(abs(P1-P3)) == 0:
        return 180

    temp = ((P2[0]-P1[0])*(P3[0]-P2[0]) + (P2[1]-P1[1])*(P3[1]-P2[1]))/(get_dist(P1,P2) * get_dist(P2,P3))

    angle = -np.arccos(min(max(temp, -1), 1))
    if np.isnan(angle):
        print('Exception:', P1, P2, P3, angle)
        angle = 0

    return abs(angle)*180/np.pi


def calc_path_cost(path):
    angle_cost = 0
    distance = 0
    for p1, p2, p3 in zip(path[0:-2], path[1:-1], path[2:]):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        angle_cost += calc_angle_cost(p1, p2, p3)
        distance += get_dist(p1, p2)

    distance += get_dist(np.array(path[-2]), np.array(path[-1]))

    return distance, angle_cost

