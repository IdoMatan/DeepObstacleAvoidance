import numpy as np
import matplotlib.pyplot as plt


def get_dist(p1, p2):
    try:
        return np.linalg.norm(p2-p1)

    except Exception:
        return np.linalg.norm(np.array(p2) - np.array(p1))


class Graph:
    def __init__(self, nodes, obstacles, smoothness):
        self.start = None
        self.end = None
        self.nodes = {}
        self.add_nodes(nodes)
        self.edges = {}
        self.n_edges = 0
        self.obstacles = obstacles
        self.smoothness = smoothness

    def add_nodes(self, nodes):
        for node in nodes:
            self.nodes[node[0]] = Vertex(*node)
            # print(f'Added node {node}')

        self.start = self.nodes[0]
        self.end = self.nodes[1000]

    def find_neighbors(self, dist=150):
        for node in self.nodes:
            for neighbor in self.nodes:
                if node != neighbor:
                    if get_dist(self.nodes[node].coordinates, self.nodes[neighbor].coordinates) <= dist:
                        name1 = str(node)+'<->'+str(neighbor)
                        name2 = str(neighbor)+'<->'+str(node)
                        if name1 not in self.edges and name2 not in self.edges:
                                self.edges[name1] = Edge(vertices=[node, neighbor],
                                                         P1=self.nodes[node].coordinates,
                                                         P2=self.nodes[neighbor].coordinates)
                        self.nodes[node].add_neighbor(neighbor)

    def find_neighbors_with_obstacles(self, dist=150):
        for node in self.nodes:
            for neighbor in self.nodes:
                if node != neighbor:

                    if get_dist(self.nodes[node].coordinates, self.nodes[neighbor].coordinates) <= dist:
                        if not self.obstacles.intercept(self.nodes[node].coordinates, self.nodes[neighbor].coordinates):
                            name1 = str(node)+'<->'+str(neighbor)
                            name2 = str(neighbor)+'<->'+str(node)
                            if name1 not in self.edges and name2 not in self.edges:
                                    self.edges[name1] = Edge(vertices=[node, neighbor],
                                                             P1=self.nodes[node].coordinates,
                                                             P2=self.nodes[neighbor].coordinates)
                            self.nodes[node].add_neighbor(neighbor)

    def plot_graph(self, topview):
        extent_x = (topview.shape[1] / 2) * self.obstacles.proportion_x
        extent_y = (topview.shape[0] / 2) * self.obstacles.proportion_y

        # plt.imshow(np.flip(topview, 0), extent=[-extent_x, extent_x, -extent_y, extent_y])
        plt.imshow(topview, extent=[-extent_x, extent_x, -extent_y, extent_y])

        # plotting edges
        for edge in self.edges:
            plt.plot([self.edges[edge].P1[0],self.edges[edge].P2[0]],[self.edges[edge].P1[1], self.edges[edge].P2[1]], '--m')

        # plotting nodes
        for node in self.nodes:
            if node == 0:
                color = 'b*'
            elif node == 1000:
                color = 'g*'
            else:
                color = 'ro'

            plt.plot(*self.nodes[node].coordinates, color)
            plt.text(*self.nodes[node].coordinates, str(self.nodes[node].name))

        plt.plot(self.obstacles.x_borders_plot, self.obstacles.y_borders_plot, '+')
        plt.show()

    def plot_graph_plan(self, plan, topview):
        extent_x = (topview.shape[1] / 2) * self.obstacles.proportion_x
        extent_y = (topview.shape[0] / 2) * self.obstacles.proportion_y
        # plt.imshow(np.flip(topview, 0), extent=[-extent_x, extent_x, -extent_y, extent_y])
        plt.imshow(topview, extent=[-extent_x, extent_x, -extent_y, extent_y])

        # plotting edges
        for edge in self.edges:
            plt.plot([self.edges[edge].P1[0],self.edges[edge].P2[0]],[self.edges[edge].P1[1], self.edges[edge].P2[1]], '--m')

        # plotting nodes
        for node in self.nodes:
            if node == 0:
                color = 'b*'
            elif node == 1000:
                color = 'g*'
            else:
                color = 'ro'

            plt.plot(*self.nodes[node].coordinates, color)
            plt.text(*self.nodes[node].coordinates, str(self.nodes[node].name))

        # plot plan
        plan_edges = []
        for node, next_node in zip(plan[:-1], plan[1:]):
            name1 = str(node) + '<->' + str(next_node)
            name2 = str(next_node) + '<->' + str(node)
            if name1 in self.edges:
                plan_edges.append(name1)
            elif name2 in self.edges:
                plan_edges.append(name2)
            else:
                print('Cant find edges:', name1, name2)

        for edge in plan_edges:
            plt.plot([self.edges[edge].P1[0], self.edges[edge].P2[0]],[self.edges[edge].P1[1], self.edges[edge].P2[1]], '-b')

        plt.show()


class Vertex:
    def __init__(self, name, x, y):
        self.name = name
        self.coordinates = np.array((x, y))
        self.neighbors = []
        self.trajectories = {}

    def get_neighbors(self):
        return self.neighbors

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def get_n_neighbors(self):
        return len(self.neighbors)


class Edge:
    def __init__(self, vertices, P1, P2, weight=1):
        self.vertices = vertices
        self.P1 = np.array(P1)
        self.P2 = np.array(P2)
        self.weight = weight
        self.length = self.get_distance()

    def get_distance(self):
        return get_dist(self.P1, self.P2)

