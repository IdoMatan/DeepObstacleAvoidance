import numpy as np
import matplotlib.pyplot as plt
from graph import *
from shapely.geometry import LineString
import cv2


def calc_angle_cost(P1,P2,P3):
    if sum(abs(P1-P3)) == 0:
        return 180

    temp = ((P2[0]-P1[0])*(P3[0]-P2[0]) + (P2[1]-P1[1])*(P3[1]-P2[1]))/(get_dist(P1,P2) * get_dist(P2,P3))

    angle = -np.arccos(min(max(temp, -1), 1))
    if np.isnan(angle):
        print('Exception:', P1, P2, P3, angle)
        angle = 0

    return abs(angle)*180/np.pi


class PathPlanner:
    def __init__(self, graph):
        self.start = graph.nodes[0]
        self.end = graph.nodes[1000]
        self.graph = graph
        self.open_vertices = []
        self.closed_vertices = []

    def calc_cost(self, node, neighbor, terminal, angle_weight=1):
        if terminal:
            return {'cost': get_dist(node.coordinates, neighbor.coordinates), 'next_node': node.name}
        else:
            min_cost = np.inf
            min_traj = None
            distance = get_dist(node.coordinates, neighbor.coordinates)
            for trajectory in node.trajectories:
                if trajectory == neighbor.name:
                    continue
                angle_cost = calc_angle_cost(neighbor.coordinates,
                                             node.coordinates,
                                             self.graph.nodes[trajectory].coordinates)

                traj_min_cost = node.trajectories[trajectory]['cost']
                cost = distance + traj_min_cost + angle_weight*angle_cost
                if cost <= min_cost:
                    min_cost = cost
                    min_traj = trajectory

            return {'cost': min_cost, 'next_node': min_traj}

    def return_min_trajectory(self, node):
        if node.name == 1000:
            return 0, 1000
        min_cost, next_node = np.inf, None
        for trajectory in node.trajectories:
            if node.trajectories[trajectory]['cost'] < min_cost:
                min_cost = node.trajectories[trajectory]['cost']
                next_node = trajectory

        return min_cost, next_node

    def get_shortest_route(self):
        terminate = False
        path = [0]
        total_cost = []
        while not terminate:
            cost, next_node = self.return_min_trajectory(self.graph.nodes[path[-1]])
            total_cost.append(cost)
            path.append(next_node)
            if next_node == 1000:
                terminate = True

        return path, total_cost

    def solve(self):
        self.open_vertices.append(self.end.name)

        while len(self.open_vertices):
            for vertice in self.open_vertices:
                if self.graph.nodes[vertice].name == 0:
                    self.open_vertices.remove(vertice)
                    continue

                for neighbor in self.graph.nodes[vertice].neighbors:
                    if self.graph.nodes[neighbor].trajectories.get(vertice):
                        continue

                    self.graph.nodes[neighbor].trajectories[vertice] =\
                        self.calc_cost(self.graph.nodes[vertice],
                                       self.graph.nodes[neighbor],
                                       terminal=True if vertice == 1000 else False)

                    self.open_vertices.append(neighbor)

                self.open_vertices.remove(vertice)

        return self.get_shortest_route()


class PathPlanningObstacles:
    def __init__(self, filename, safety_factor=0, proportion_x=1, proportion_y=1):
        self.safety_factor = safety_factor
        self.x_borders = []
        self.x_borders_plot = []
        self.y_borders = []
        self.y_borders_plot = []
        self.proportion_x = proportion_x
        self.proportion_y = proportion_y
        self.number = 0
        self.obstacles = self.find_corners(filename)
        self.obstacles = self.obstacles[1:]

        self.borders()

    def find_corners(self, filename):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img1 = cv2.imread('airsim_drone/real_190.png')

        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        obstacles = []
        self.number = contours.__len__() - 1
        for contour in contours:

            extent_x = (gray.shape[1] / 2) * self.proportion_x
            extent_y = (gray.shape[0] / 2) * self.proportion_y
            # plt.imshow(np.flip(gray, 0), extent=[-extent_x, extent_x, -extent_y, extent_y])
            # plt.imshow(gray, extent=[-extent_x, extent_x, -extent_y, extent_y])
            plt.imshow(img1, extent=[-extent_x, extent_x, -extent_y, extent_y])

            min_x = gray.shape[1]; max_x = 0
            min_y = gray.shape[0]; max_y = 0
            for contou in contour:
                img[contou[0][1], contou[0][0]] = [0, 0, 255]
                min_x = contou[0][0] if min_x > contou[0][0] else min_x
                max_x = contou[0][0] if max_x < contou[0][0] else max_x
                min_y = contou[0][1] if min_y > contou[0][1] else min_y
                max_y = contou[0][1] if max_y < contou[0][1] else max_y

            # Centering the obstacle
            O = np.array([(min_x - gray.shape[1] / 2) * self.proportion_x, (144 - max_y - gray.shape[0] / 2) * self.proportion_y,
                          (max_x - gray.shape[1] / 2) * self.proportion_x, (144 - min_y - gray.shape[0] / 2) * self.proportion_y])
            obstacles.append(O)
            plt.plot((min_x - gray.shape[1] / 2) * self.proportion_x, (144 - max_y - gray.shape[0] / 2) * self.proportion_y, '+')
            plt.plot((max_x - gray.shape[1] / 2) * self.proportion_x, (144 - min_y - gray.shape[0] / 2) * self.proportion_y, '+')

        # plt.show()

        # cv2.imwrite(os.path.normpath(os.path.join(tmp_dir, "corners" + '.png')), img)
        return obstacles

    def update_safety_factor(self, safety_factor):
        self.safety_factor = safety_factor
        self.borders()

    def borders(self):
        self.x_borders = []
        self.y_borders = []

        for obstacle in self.obstacles:
            self.x_borders.append(range(int(obstacle[0])-self.safety_factor, int(obstacle[2])-self.safety_factor))
            # r = range(int(obstacle[0])-self.safety_factor, int(obstacle[2])-self.safety_factor)
            # self.x_borders.append([*r])
            # self.x_borders += [*r]
            self.x_borders_plot += [obstacle[0], obstacle[2]]

            self.y_borders.append(range(int(obstacle[1])+self.safety_factor, int(obstacle[3])+self.safety_factor))
            # r_y = range(int(obstacle[1])+self.safety_factor, int(obstacle[3])+self.safety_factor)
            # self.x_borders.append([*r_y])
            # self.y_borders += [*r_y]
            self.y_borders_plot += [obstacle[1], obstacle[3]]

    def intercept(self, point1, point2):
        line1 = LineString([(point1[0], point1[1]), (point2[0], point2[1])])
        for obstacle in self.obstacles:
            line2 = LineString([(obstacle[0], obstacle[1]), (obstacle[2], obstacle[3])])
            if not line1.intersection(line2).is_empty:
                return True
            line2 = LineString([(obstacle[0], obstacle[3]), (obstacle[2], obstacle[1])])
            if not line1.intersection(line2).is_empty:
                return True
        return False


def generate_nodes(map_lim, obstacles, x_init, x_goal, n_points=100):
    nodes = []

    for point in range(1, n_points):
        in_freespace = True
        while in_freespace:
            x_node = np.random.randint(map_lim[0], map_lim[1])
            y_node = np.random.randint(map_lim[2], map_lim[3])
            # x_node = int(max(min(np.random.normal(0, 100, 1), 200), -200))
            # y_node = int(max(min(np.random.normal(0, 100, 1), 200), -200))

            in_freespace = any([x_node in obstacles.x_borders[i] for i in range(obstacles.number)] and
                               [y_node in obstacles.y_borders[i] for i in range(obstacles.number)])

        nodes.append((point, x_node, y_node))

    nodes.append((0, x_init[0], x_init[1]))
    nodes.append((1000, x_goal[0], x_goal[1]))

    return nodes


def path_planning(map_lim=[-128, 128, -72, 72],  # map_lim=np.array([(-500, 500), (-200, 200)]),  #  dimensions of Search Space
                  x_init=(0, 0),                                # starting location
                  x_goal=(0, 90),                            # goal location
                  n_points=50,                               # Number of random nodes
                  obstacles=None,                                 # obstacles
                  safety_factor=0,                              # length of smallest edge to check for intersection with obstacles
                  smoothness=32,                                # smoothness
                  dist=150,
                  topview=None,
                  plot=True,
                  method='smooth_rrt'  # 'smooth_rrt'
                  ):

    if method == 'smooth':
        # obstacles = PathPlanningObstacles(obstacles, safety_factor)
        obstacles.update_safety_factor(safety_factor)

        nodes = generate_nodes(map_lim=map_lim, obstacles=obstacles, x_init=x_init, x_goal=x_goal, n_points=n_points)

        graph = Graph(nodes, obstacles, smoothness)
        graph.find_neighbors_with_obstacles(dist=dist)
        graph.plot_graph(topview)

        planner = PathPlanner(graph)
        plan, cost = planner.solve()

        graph.plot_graph_plan(plan, topview)

        print('Plan:', plan)
        print('Costs:', cost)
        print('Total Cost:', cost[0])
        return plan

    if method == 'smooth_rrt':

        from rrt_src.rrt.rrt_smooth import RRTSmooth
        from rrt_src.search_space.search_space import SearchSpace
        from rrt_src.utilities.plotting import Plot
        from rrt_src.utilities.obstacle_generation import obstacle_generator
        from rrt_src.utilities.geometry import calc_path_cost

        # create Search Space
        X = SearchSpace(np.array([(map_lim[0], map_lim[1]), (map_lim[2], map_lim[3])]))
        obstacle_generator(obstacles.obstacles, X)

        max_allowed_angle = 35
        Q = np.array([(8, 4, 5)])  # length of tree edges
        r = 1  # length of smallest edge to check for intersection with obstacles
        max_samples = 5024  # max number of samples to take before timing out
        prc = 0.1  # probability of checking for a connection to goal

        # create rrt_search
        best_path = None
        best_cost = 100000
        best_distance = 100000
        rrt = None

        for episode in range(5):

            rrt = RRTSmooth(X, Q, x_init, x_goal, max_samples, r, theta_tol=max_allowed_angle, prc=prc)
            path = rrt.rrt_search()
            if path is not None:
                length, cost = calc_path_cost(path)
                print(f'Episode {episode}, path length: {length}, angle cost: {cost}')
                if cost <= best_cost:
                    best_path = path
                    best_cost = cost
                    best_distance = length

        # plot
        plot = Plot("rrt_2d_with_random_obstacles")
        plot.plot_tree(X, rrt.trees)
        if best_path is not None:
            # length, cost = calc_path_cost(path)
            print(f'Best path length: {best_distance}, Best angle cost: {best_cost}')
            plot.plot_path(X, best_path)
        plot.plot_obstacles(X, obstacles.obstacles)
        plot.plot_start(X, x_init)
        plot.plot_goal(X, x_goal)
        plot.draw(auto_open=True)
        return path, rrt
