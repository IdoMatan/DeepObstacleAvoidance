from src.rrt.rrt_base import RRTBase
from src.utilities.geometry import calc_angle_cost
import numpy as np


class RRTSmooth(RRTBase):
    def __init__(self, X, Q, x_init, x_goal, max_samples, r, theta_tol=360, prc=0.01):
        """
        Template RRT planner
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        super().__init__(X, Q, x_init, x_goal, max_samples, r, prc)
        self.theta_tol = theta_tol

    def connect_to_point_wrt_angle(self, tree, x_a, x_b):
        """
        Connect vertex x_a in tree to vertex x_b
        :param tree: int, tree to which to add edge
        :param x_a: tuple, vertex
        :param x_b: tuple, vertex
        :return: bool, True if able to add edge, False if prohibited by an obstacle
        """
        angle = 0

        try:
            if self.trees[tree].V_count > 1:
                grand_parent = np.array((self.trees[tree].E[x_a][0], self.trees[tree].E[x_a][1]))
                parent = np.array(x_a)
                child = np.array(x_b)

                angle = calc_angle_cost(grand_parent, parent, child)
                # print('angle:', angle)
        except:
            print(" Execption ---- > can't get angle")

        if self.trees[tree].V.count(x_b) == 0 and self.X.collision_free(x_a, x_b, self.r) and angle < self.theta_tol:
            self.add_vertex(tree, x_b)
            self.add_edge(tree, x_b, x_a)
            return True
        return False


    def rrt_search(self):
        """
        Create and return a Rapidly-exploring Random Tree, keeps expanding until can connect to goal
        https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
        :return: list representation of path, dict representing edges of tree in form E[child] = parent
        """
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        while True:
            for q in self.Q:  # iterate over different edge lengths until solution found or time out
                for i in range(q[1]):  # iterate over number of edges of given length to add
                    x_new, x_nearest = self.new_and_near(0, q)

                    if x_new is None:
                        continue

                    # connect shortest valid edge
                    self.connect_to_point_wrt_angle(0, x_nearest, x_new)

                    solution = self.check_solution()
                    if solution[0]:
                        return solution[1]
