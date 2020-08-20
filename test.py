import numpy as np


def get_dist(p1, p2):
    try:
        return np.linalg.norm(p2-p1)
    except Exception:
        return np.linalg.norm(np.array(p2) - np.array(p1))


def calc_angle_cost(P1,P2,P3):
    temp = ((P2[0]-P1[0])*(P3[0]-P2[0]) + (P2[1]-P1[1])*(P3[1]-P2[1]))/(get_dist(P1,P2) * get_dist(P2,P3))

    angle = -np.arccos(min(max(temp, -1), 1))

    # angle = -np.arccos(((P2[0]-P1[0])*(P3[0]-P2[0]) + (P2[1]-P1[1])*(P3[1]-P2[1]))/(get_dist(P1,P2) * get_dist(P2,P3)))
    return abs(angle)*180/np.pi

P1 = np.array((32, 232))
P2 = np.array((38, 280))
P3 = np.array((40, 296))

print(calc_angle_cost(P1,P2,P3)*180/np.pi)
