from fractions import Fraction
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from time import time
import os
import pickle



class Triangle:
    '''
    Create a triangle for measuring distance between an edge and a node.
    Nodes/vertices are labeled A, B, and C. The edges between nodes are
    labeled based on the end nodes (i.e., AB). The angles between edges
    are labeled based on the included edges. Angles are measured in
    degrees.
    '''
    def __init__(self, edge, node):
        self.A = edge.node1
        self.B = edge.node2
        self.C = node
        self.AB = edge
        self.CA = Edge(self.C, self.A)
        self.BC = Edge(self.B, self.C)
        self.ABC = self.find_angle(self.AB, self.BC)
        self.BCA = self.find_angle(self.BC, self.CA)
        self.CAB = self.find_angle(self.CA, self.AB)
        self.lengthAB = self.AB.find_length()
        self.lengthCA = self.CA.find_length()
        self.lengthBC = self.BC.find_length()
        # self.draw_triangle()

    def find_angle(self, edge1, edge2):
        '''
        Find angle between edge1 and edge2 and return it. Angle is
        measured in degrees. If angle is negative, add 180 degrees
        to make it positive.
        '''
        angle = math.degrees(math.atan(edge1.m - edge2.m)/(1.0 + edge1.m*edge2.m))
        if angle < 0.0:
            angle += 180.0
        return angle

    def find_min_distance(self):
        '''
        Find the minimum distance between an edge and a node. The edge is AB;
        the node is C. If angles ABC or CAB are greater than 90 degrees, then
        use the corresponding edge's length as minimum distance. Otherwise,
        use the following formula to find the length of the perpendicular line
        between AB and C:
        min_distance = abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1)) /
                       sqrt((x2 - x1)^2 + (y2 - y1)^2)
        where
            (x1, y1) are the coordinates for A
            (x2, y2) are the coordinates for B
            (x0, y0) are the coordinates for C
        '''
        min_distance = 0.0
        if (self.ABC > 90.0) or (self.CAB > 90.0):
            if self.lengthCA < self.lengthBC:
                min_distance = self.lengthCA
            else:
                min_distance = self.lengthBC
        else:
            x_coord0 = self.C.coords[0]
            y_coord0 = self.C.coords[1]
            x_coord1 = self.A.coords[0]
            y_coord1 = self.A.coords[1]
            x_coord2 = self.B.coords[0]
            y_coord2 = self.B.coords[1]
            numerator = abs((x_coord2 - x_coord1)*(y_coord1 - y_coord0) - (x_coord1 - x_coord0)*(y_coord2 - y_coord1))
            denominator = math.sqrt((x_coord2 - x_coord1)**2 + (y_coord2 - y_coord1)**2)
            min_distance = numerator/denominator
        return min_distance

    def draw_triangle(self):
        '''
        Draw Triangle object using NetworkX. (Used for debugging and documentation only.)
        '''
        _, ax = plt.subplots()  # Axes object required to add axes to Networkx graph

        G = nx.Graph()  # create a Directed Graph object

        # add each Node to Graph
        G.add_node('A', pos=self.A.coords)
        G.add_node('B', pos=self.B.coords)
        G.add_node('C', pos=self.C.coords)

        # add each Edge to Graph
        G.add_edge('A', 'B')
        G.add_edge('C', 'A')
        G.add_edge('B', 'C')

        nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, ax=ax)

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_axis_on()
        plt.title("Triangle for Calculating Distance")
        plt.show()

class Node:
    '''
    Represent a point in 2D-space by its name ("label") and coordinates ("coords").
    '''
    def __init__(self, label, coords):
        self.label = label
        self.coords = coords

class Edge:
    '''
    Represent an edge in 2D-space by the two nodes that comprise it, its slope,
    and its length. Include the closest node to it and distance between edge and
    node for nearest-edge insertion heuristic.
    '''
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.m = self.find_slope(node1, node2)
        self.length = self.find_length()
        self.closest_node = None
        self.closest_node_dist = math.inf

    def find_slope(self, node1, node2):
        '''
        Find the slope of the edge using:
        m = (y2 - y1) / (x2 - x1)
        '''
        x_coord1 = node1.coords[0]
        y_coord1 = node1.coords[1]
        x_coord2 = node2.coords[0]
        y_coord2 = node2.coords[1]
        m = (y_coord2 - y_coord1)/(x_coord2 - x_coord1)
        return m

    def find_length(self):
        '''
        Find length of edges using the following formula:
        length = sqrt((x2 - x1)^2 + (y2 - y1)^2)
        '''
        x_coord1 = self.node1.coords[0]
        y_coord1 = self.node1.coords[1]
        x_coord2 = self.node2.coords[0]
        y_coord2 = self.node2.coords[1]
        return math.sqrt((x_coord2 - x_coord1)**2 + (y_coord2 - y_coord1)**2)

    def reset_edge(self):
        '''
        Reset edge to select a new closest node and distance to it.
        '''
        self.closest_node = None
        self.closest_node_dist = math.inf

class Network:
    '''
    Represent a collection of nodes in 2D-space. Track nodes available for
    insertion and nodes already inserted for nearest-edge insertion heuristic.
    Store the optimal tour (list of nodes ordered by insertion), the tour's cost
    (sum of edge lengths), and the edges that comprise the network.
    '''
    def __init__(self, coords):
        '''
        Take the coords and build a Network of Nodes from them.
        '''
        self.nodes = {}  # Nodes in the Network (does not change once initialized)
        node_labels = list(range(1, len(coords)+1))
        for coord_pair in coords:
            label = node_labels.pop(0)
            node = Node(label, coord_pair)
            self.nodes[label] = node  

        self.available_nodes = self.nodes.copy()  # Nodes available for insertion
        self.nodes_in_tour = []  # Nodes in order of insertion
        self.edges_in_tour = []  # Edges in order of insertion
        self.cost = 0 # total cost of Tour (sum of Edges' lengths)
        self.optimal_nodes = None  # list of Nodes in order of insertion for optimal tour
        self.optimal_edges = None  # list of Edges in order of insertion for optimal tour
        self.optimal_cost = math.inf  # cost of optimal tour

    def add_node_to_tour(self, node):
        '''
        Pop Node from dictionary of Nodes by key and append it to nodes_in_tour
        to indicate insertion.
        '''
        self.nodes_in_tour.append(self.available_nodes.pop(node.label))

    def add_edge_to_tour(self, node1, node2):
        '''
        Append Edge to edges_in_tour to indicate insertion.
        '''
        self.edges_in_tour.append(Edge(node1, node2))

    def find_distance(self, node1, node2):
        ''' Find cost (as Euclidean distance) using the following formula:
        cost = sqrt((x2 - x1)^2 + (y2 - y1)^2)
        '''
        coords1_x, coords1_y = node1.coords
        coords2_x, coords2_y = node2.coords
        x_diff_squared = (coords2_x - coords1_x)**2
        y_diff_squared = (coords2_y - coords1_y)**2
        return math.sqrt(x_diff_squared + y_diff_squared)

    def find_first_edge(self, starting_node):
        '''
        Find shortest distance between starting_node and available Nodes
        to create and insert the first edge in the tour.
        '''
        self.add_node_to_tour(starting_node)
        min_distance = math.inf
        closest_node_key = None
        for node in self.available_nodes.values():
            distance = self.find_distance(starting_node, node)
            if distance < min_distance:
                min_distance = distance
                closest_node_key = node.label
        closest_node = self.available_nodes[closest_node_key]  # get closest node from available_nodes
        self.add_node_to_tour(closest_node)
        self.add_edge_to_tour(starting_node, closest_node)

    def tiebreaker(self, edge1, edge2):
        '''
        Select the Edge that when removed and its Nodes are incorporated into
        new Edges increases tour cost (sum of Edge lengths) less.
        '''
        current_cost = 0
        for edge in self.edges_in_tour:
            current_cost += edge.length
        cost_minus_edge1 = current_cost - edge1.length
        cost_minus_edge2 = current_cost - edge2.length
        edge1AddedLength = Edge(edge1.node1, edge1.closest_node).length
        edge1AddedLength += Edge(edge1.node2, edge1.closest_node).length
        edge2AddedLength = Edge(edge2.node1, edge2.closest_node).length
        edge2AddedLength += Edge(edge2.node2, edge2.closest_node).length
        edge1_cost = cost_minus_edge1 + edge1AddedLength
        edge2_cost = cost_minus_edge2 + edge2AddedLength
        if edge1_cost <= edge2_cost:
            return edge1
        else:
            return edge2

    def iteration(self):
        '''
        Helper function to implement closest edge insertion heuristic.
        Tracks time to execute heuristic and iterates over all possible
        starting nodes to find the optimal tour (tour with min cost).
        '''
        for i in range(1, len(self.nodes)+1):
            self.insertion(self.available_nodes[i])
            if self.cost < self.optimal_cost:
                self.optimal_cost = self.cost
                self.optimal_nodes = self.nodes_in_tour.copy()
                self.optimal_edges = self.edges_in_tour.copy()
            self.available_nodes = self.nodes.copy()
            self.nodes_in_tour = []
            self.edges_in_tour = []
            self.cost = 0

    def insertion(self, starting_node):
        '''
        Function to implement closest edge insertion heuristic.
        Implements heuristic for a single tour based on starting_node.
        '''
        self.find_first_edge(starting_node)
        optimal_edge = None
        i = 1
        while(len(self.available_nodes) > 0):
            for edge in self.edges_in_tour:
                for node in self.available_nodes.values():
                    triangle = Triangle(edge, node)
                    min_distance = triangle.find_min_distance()
                    if min_distance < edge.closest_node_dist:
                        edge.closest_node_dist = min_distance
                        edge.closest_node = node
                if optimal_edge:
                    if edge.closest_node_dist < optimal_edge.closest_node_dist:
                        optimal_edge = edge
                    elif edge.closest_node_dist == optimal_edge.closest_node_dist:
                        optimal_edge = self.tiebreaker(edge, optimal_edge)
                else:
                    optimal_edge = edge

            closest_node = optimal_edge.closest_node
            self.add_node_to_tour(closest_node)
            node1 = optimal_edge.node1
            node2 = optimal_edge.node2
            self.edges_in_tour.append(Edge(node1, closest_node))
            self.edges_in_tour.append(Edge(node2, closest_node))
            if (len(self.nodes_in_tour) > 3):
                self.edges_in_tour.remove(optimal_edge)

            for edge in self.edges_in_tour:
                edge.reset_edge()
            optimal_edge = None
        for edge in self.edges_in_tour:
            self.cost += edge.length

    def plot_route(self, filepath):
        '''
        Plot Nodes and Edges of Network based on their coordinates. Save PNG image
        under Results folder.
        '''
        _, ax = plt.subplots()  # require Axes object to add axes to Networkx graph

        G = nx.Graph()  # create a Directed Graph object

        # add each Node to Graph
        for node in self.optimal_nodes:
            G.add_node(node.label, pos=node.coords)

        # add each Edge to Graph
        for edge in self.optimal_edges:
            G.add_edge(edge.node1.label, edge.node2.label)

        node_sizes = [60] * len(self.optimal_nodes)
        node_colors = ['Orange'] * len(self.optimal_nodes)
        nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=node_sizes,
                node_color=node_colors, ax=ax)

        nx.draw_networkx_labels(G, nx.get_node_attributes(G, 'pos'), font_size=6)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_axis_on()
        filename = filepath.split(".")[0]
        plt.title(f"Optimal Tour for {filename}")
        plt.savefig(f"Results/{filename}_Soln.png", bbox_inches="tight")
        plt.show()

    def write_to_csv(self, filepath, running_time):
        '''
        Write results of closest-edge insertion heuristic to a CSV file.
        '''
        filename = filepath.split(".")[0]
        optimal_nodes = [node.label for node in self.optimal_nodes]
        optimal_edges = []
        for edge in self.optimal_edges:
            optimal_edges.append((edge.node1.label, edge.node2.label))
        results_list = [[filename, optimal_nodes, optimal_edges, self.optimal_cost, round(running_time, 6)]]
        columns = columns=['TSP File', 'Optimal Tour', 'Edges (in order of insertion)', 'Cost', 'Running Time (sec)']
        results = pd.DataFrame(results_list, columns=columns)
        with open(f'Results/{filename}.csv', 'w', newline='') as file:
            file.write(results.to_csv(index=False))

def parse_coords(tsp_file):
    ''' Parse coordinates from TSP file and return them as tuple pairs.'''
    f = open(tsp_file, "r")
    lines = f.readlines()
    size = len(lines)
    lines = lines[7:size]  # skips first seven lines of metadata
    coords = []
    for line in lines:
        _, coord_x, coord_y = line.replace("\n", "").split(" ")
        coords.append((float(coord_x), float(coord_y)))
    return coords

if __name__ == "__main__":
    os.system("cls")
    tsp_filepath = input("Enter .tsp filename to solve:\n> ")
    os.system("cls")
    if not os.path.exists("Results"):
        os.makedirs("Results")  # create Results folder if one does not exist
    coords = parse_coords(tsp_filepath)  # parse TSP file for coordinates
    network = Network(coords)
    start_time = time()
    network.iteration()
    running_time = time() - start_time
    network.plot_route(tsp_filepath)
    network.write_to_csv(tsp_filepath, running_time)
