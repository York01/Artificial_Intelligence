import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from time import time


class City:
    '''
    Represent a city for mapping a route. It includes city-specific
    data (coords) and details to build a route (prev_city).
    '''
    def __init__(self, num, coords):
        self.num = num  # city number (1, 2, 3...)
        self.not_visited = True  # Avoid visitng City more than once to avoid cycles
        self.coords = coords  # xy-coordinates of the city
        self.connections = []  # cities the current one has a direct connection to
        self.costs = []  # cost to visit each connection
        self.prev_city = None  # city visited previously
        self.curr_cost = 0  # cost to visit this city based on current route

    def add_connections(self, connections):
        '''
        Adds direct connections and calculate cost to visit each.
        '''
        self.connections += connections
        for connection in self.connections:
            self.costs.append(find_cost(self.coords, connection.coords))

class Map:
    '''
    Represents a map to track which cities are mapped, the order in which to
    visit them (based on algorithm), and the optimal route/cost.
    '''
    def __init__(self, cities):
        self.cities = cities  # Cities in the Map object
        self.route = None  # optimal route
        self.cost = math.inf  # cost for optimal route

    def reset(self):
        '''
        Resets City and Map attributes. Ensures each algorithm runs
        without prior runs affecting its outcome.
        '''
        for city in self.cities:
            city.not_visited = True
            city.prev_city = None
            city.curr_cost = 0
        self.route = []
        self.cost = math.inf

    def BFS(self, start_city):
        '''
        Performs breadth-first search on a Map object to find the optimal
        route from the start_city to city 11.
        '''
        start_time = time()
        self.reset()
        print("BFS Traveling Salesman Problem")
        to_visit = []  # List object to be used a first in, first out queue
        to_visit.insert(0, start_city)
        while(len(to_visit) > 0):
            curr_city = to_visit.pop(0)  # dequeue next City to visit
            sorted_connections = sort_by_cost(curr_city)
            for next_city in sorted_connections:  # visit Cities in order of lowest cost
                if next_city.not_visited:
                    next_city.not_visited = False  # avoid visiting Cities more than once
                    next_city.prev_city = curr_city  # track previous city to reconstruct route
                    new_cost = find_cost(curr_city.coords, next_city.coords)
                    next_city.curr_cost += curr_city.curr_cost + new_cost
                    to_visit.append(next_city)  # enqueue next City to visit

            if curr_city == self.cities[10]:
                    if curr_city.curr_cost < self.cost:
                        curr_route = get_route(curr_city)
                        print(f"Route: {curr_route} with cost {round(curr_city.curr_cost, 6)}\n"
                              f"is better than {self.route} with cost {round(self.cost, 6)}\n")
                        self.cost = curr_city.curr_cost
                        self.route = curr_route
        print(f"Optiaml route for BFS: {self.route} with a cost of {round(self.cost, 6)}\n")
        return time() - start_time

    def DFS(self, start_city):
        '''
        Performs depth-first search on a Map object to find the optimal
        route from the start_city to city 11.
        '''
        start_time = time()
        self.reset()
        print("DFS Traveling Salesman Problem")

        self.DFS_visit(start_city)
        print(f"Optiaml route for DFS: {self.route} with a cost of {round(self.cost, 6)}\n")
        return time() - start_time

    def DFS_visit(self, curr_city):
        '''
        Visits Cities belonging to a Map object as part of DFS.
        '''
        new_cost = 0
        sorted_connections = sort_by_cost(curr_city)
        for next_city in sorted_connections:  # visit Cities in order of lowest cost
            if next_city.not_visited:
                next_city.not_visited = False  # avoid visiting Cities more than once
                next_city.prev_city = curr_city  # track previous city to reconstruct route
                new_cost = find_cost(curr_city.coords, next_city.coords)
                next_city.curr_cost += curr_city.curr_cost + new_cost
                self.DFS_visit(next_city)
        if curr_city == self.cities[10]:
                if curr_city.curr_cost < self.cost:
                    curr_route = get_route(curr_city)
                    print(f"Route: {curr_route} with cost {round(curr_city.curr_cost, 6)}\n"
                          f"is better than {self.route} with cost {round(self.cost, 6)}\n")
                    self.cost = curr_city.curr_cost
                    self.route = curr_route

def plot_route(coords, route, cities, algorithm):
    '''Plot city coordinates in the order specified by route and save PNG image
    under Results.'''
    sorted_coords = []
    _, ax = plt.subplots()  # require Axes object to add axes to Networkx graph

    G = nx.DiGraph()  # create a Directed Graph object

    # add each City as a Node to the graph
    for city in cities:
        G.add_node(city.num, pos=city.coords)

    # add an orange edge for each connection between Cities in route
    cities_in_route = range(len(route)-1)
    for city in cities_in_route:
        G.add_edge(route[city], route[city+1], color="orange")

    # add a black edge for each remaining connection between Cities
    for city in cities:
        for connection in city.connections:
            if not G.has_edge(city.num, connection.num):
                G.add_edge(city.num, connection.num, color="black")

    # retrieve edge colors to pass to draw_networkx()
    colors = nx.get_edge_attributes(G,'color').values()
    nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, edge_color=colors, ax=ax)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_axis_on()

    plt.title(f"Optimal Route with {algorithm}")
    plt.savefig(f"Results/{algorithm}_Soln.png", bbox_inches="tight")


def get_route(curr_city):
    '''
    Reconstruct route from each city's previous city.
    '''
    route = [curr_city.num]
    while(curr_city.prev_city):
        curr_city = curr_city.prev_city
        route.append(curr_city.num)
    return route[::-1]  # reverses order


def sort_by_cost(city):
    '''
    Sort a city's connections by the cost to visit them from low to high.
    '''
    to_sort = zip(city.costs, city.connections)
    return [x for y, x in sorted(to_sort)]

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

def find_cost(coords1, coords2):
    ''' Find cost (as Euclidean distance) using the following formula:
    cost = sqrt((x2 - x1)^2 + (y2 - y1)^2)'''
    coords1_x, coords1_y = coords1
    coords2_x, coords2_y = coords2
    x_diff_squared = (coords2_x - coords1_x)**2
    y_diff_squared = (coords2_y - coords1_y)**2
    return round(math.sqrt(x_diff_squared + y_diff_squared), 6)

def map_cities(coords):
    '''
    Takes the set of coordinates passed as an argument, creates a City
    object from each, adds connections for each city (hard-coded), and
    creates a Map object from the cities.
    '''
    cities = []  # list of City objects to map
    city_names = list(range(1, len(coords)+1))  # build list of numbers from 1 to 11
    for pair in coords:
        city = City(city_names.pop(0), pair)  # create a City object for each coordinate pair
        cities.append(city)

    # Zero-based indexing - add 1 to numbers below to determine cities
    # i.e., city 1 has a direct connection to cities 2, 3, and 4
    cities[0].add_connections([cities[1], cities[2], cities[3]])
    cities[1].add_connections([cities[2]])
    cities[2].add_connections([cities[3], cities[4]])
    cities[3].add_connections([cities[4], cities[5], cities[6]])
    cities[4].add_connections([cities[6], cities[7]])
    cities[5].add_connections([cities[7]])
    cities[6].add_connections([cities[8], cities[9]])
    cities[7].add_connections([cities[8], cities[9], cities[10]])
    cities[8].add_connections([cities[10]])
    cities[9].add_connections([cities[10]])
    return Map(cities)  # create Map from list of City objects

def write_to_csv(results):
    ''' Convert DataFrame to CSV format and write it to a file under
    Results.'''
    with open(f'Results/11PointBFSDFS.csv', 'w', newline='') as file:
        file.write(results.to_csv(index=False))

if __name__ == "__main__":
    os.system("cls")
    tsp_filepath = input("Enter path of .tsp file to solve using BFS and DFS.\n> ")
    os.system("cls")
    if not os.path.exists("Results"):
        os.makedirs("Results")  # creates a Results folder if one does not exist

    # parse city xy-coordinates and map cities based on them
    coords = parse_coords(tsp_filepath)  # get city coordinates from a single TSP file
    city_map = map_cities(coords)

    # Runs BFS and DFS to find optimal routes based on map
    results = []
    algorithms = ["BFS", "DFS"]
    for algorithm in algorithms:
        running_time = 0
        if algorithm == "BFS":
            running_time = city_map.BFS(city_map.cities[0])
        elif algorithm == "DFS":
            running_time = city_map.DFS(city_map.cities[0])
        results.append([algorithm, city_map.route, round(city_map.cost, 6), running_time])
        plot_route(coords, city_map.route, city_map.cities, algorithm)

    # create a DataFrame from the results and use pandas built-in function to write it to a CSV file in Results
    results_df = pd.DataFrame(results, columns=["Algorithm", "Optimal Route", "Cost", "Running Time (sec)"])
    write_to_csv(results_df)