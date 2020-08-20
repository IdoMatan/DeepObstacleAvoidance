import numpy as np
import matplotlib.pyplot as plt
from computer_vision.path_planner import *
from computer_vision.graph import *


class PathPlanningObstacles:

    def __init__(self, obstacles):
        self.obstacles = obstacles

    def borders(self):
        for obstacle in self.obstacles:
            self.

def create_nodes(map, obstacles, x_init, x_goal, n_points=100):
    nodes = []

    for point in range(1, n_points):
        in_freespace = False
        while in_freespace:
            x_node = np.random.randint(map[0], map[1])
            y_node = np.random.randint(map[2], map[3])
            in_freespace = True if x_node in obstacles.freespsace_x and y_node in obstacles.freespace_y else False

        nodes.append((point, x_node, y_node))

    nodes.append((0, x_init[0], x_init[1]))
    nodes.append((1000, x_goal[0], x_goal[1]))

    return nodes


map_limits = [0, 300, 0, 300]
nodes = create_nodes(map_limits, 500)

graph = Graph(nodes)
graph.find_neighbors(dist=60)
graph.plot_graph()

planner = PathPlanner(graph)
plan, cost = planner.solve()

graph.plot_graph_plan(plan)

print('Plan:', plan)
print('Costs:', cost)
print('Total Cost:', cost[0])



def path_planning(X_dimensions = np.array([(-500, 500), (-200, 200)]),  #  X_dimensions = np.array([(-255, 255), (-145, 145)]),  # dimensions of Search Space
                   x_init=(0, 0),                                  # starting location
                   x_goal=(100, 100),                              # goal location
                   n_points = 100,
                   obstacles=[],
                   Q = np.array([(8, 4)]),                         # length of tree edges
                   r=1,                                            # length of smallest edge to check for intersection with obstacles
                   max_samples=1024,                               # max number of samples to take before timing out
                   rewire_count=32,                                # optional, number of nearby branches to rewire
                   prc=0.1,                                        # probability of checking for a connection to goal):
                   plot=True,
                   ):

    create_nodes(map=X_dimensions, x_init, x_goal, n_points, obstacles)

    map_limits = [0, 300, 0, 300]
    nodes = create_nodes(map_limits, 500)

    graph = Graph(nodes)
    graph.find_neighbors(dist=60)
    graph.plot_graph()

    planner = PathPlanner(graph)
    plan, cost = planner.solve()

    graph.plot_graph_plan(plan)

    print('Plan:', plan)
    print('Costs:', cost)
    print('Total Cost:', cost[0])
