import os
from time import time
import math
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_route(coords, route, filename):
    '''Plot city coordinates in the order specified by route and save PNG image
    under Results.'''
    sorted_coords = []

    # sort coordinates based on the route
    for city in route:
        sorted_coords.append(coords[city-1])

    df = pd.DataFrame(sorted_coords)

    x = df.iloc[:, 0].values  # convert x- and y-components to a numpy array
    y = df.iloc[:, 1].values

    x_diff = np.diff(x)  # calculate the distance between x- and y- components
    y_diff = np.diff(y)
    print(x_diff)

    pos_x = x[:-1] + x_diff/2  # positions to draw arrows from (halfway point)
    pos_y = y[:-1] + y_diff/2
    norm = np.sqrt(x_diff**2+y_diff**2)  # get norms for finding unit vectors

    dir_x = x_diff/norm  # get the x- and y-components of the unit vectors
    dir_y = y_diff/norm

    _, ax = plt.subplots()  # allow subplots on the same axes
    ax.plot(x,y, marker="o")  # plot coordinates with a line connecting them

    # overlay starting coordinate with an orange marker
    ax.plot(x[0], y[0], marker="o", color="orange")
    # plot arrows between coordinates in order of the route
    ax.quiver(pos_x, pos_y, dir_x, dir_y, angles="xy", pivot="mid", zorder=3)

    plt.title(f"Optimal Route for {filename}")
    plt.savefig(f"Results/{filename}_Soln.png", bbox_inches="tight")

def write_to_csv(results) -> None:
    ''' Convert DataFrame to CSV format and write it to a file under
    Results.'''
    with open(f'Results/TSP_results.csv', 'w', newline='') as file:
        file.write(results.to_csv(index=False))

def find_cost(coords1, coords2):
    ''' Find cost (as Euclidean distance) using the following formula:
    cost = sqrt((x2 - x1)^2 + (y2 - y1)^2)'''
    coords1_x, coords1_y = coords1
    coords2_x, coords2_y = coords2
    x_diff_squared = (coords2_x - coords1_x)**2
    y_diff_squared = (coords2_y - coords1_y)**2
    return round(math.sqrt(x_diff_squared + y_diff_squared), 6)

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

def get_cost_matrix(coords):
    ''' Create a 2D numpy array (matrix) of the cost to travel between
    coordinates.'''
    matrix = []
    for coord1 in coords:
        row = []
        for coord2 in coords:
            row.append(find_cost(coord1, coord2))
        matrix.append(np.array(row))
    return np.array(matrix)
    
def solve_TSP(coords, filename):
    '''Find the optimal route (minimum cost) to visit all cities and
    return to starting city.'''
    filename = filename.split(".")[0]
    cost_matrix = get_cost_matrix(coords)
    min_cost = math.inf  # whatever the first cost is will be less than infinity
    optimal_route = None

    cities = list(range(1, len(coords)+1))
    routes = itertools.permutations(cities)  # return an iterator for all routes
    for route in routes:
        route = list(route)
        route.append(route[0])  # include returning to starting city in cost
        cost = 0
        for city in range(0, len(route)-1):
            cost += cost_matrix[route[city]-1, route[city+1]-1]
        cost = round(cost, 6)
        if cost < min_cost:
            min_cost = cost
            optimal_route = route
    print(f"The optimal route for {filename}: {optimal_route}\nCost: {min_cost}\n")
    plot_route(coords, optimal_route, filename)
    return optimal_route, min_cost

def get_file_number(route):
    return int(route.split("Random")[1].split(".tsp")[0])

if __name__ == "__main__":
    os.system("cls")
    mode = input("Enter 1 if you want to solve for a single TSP file and plot the route.\n"
                 "Enter 2 if you want to solve for all TSP files in a given"
                 " directory, plot each route, and write the results to a CSV file.\n> ")
    if mode != "1" and mode != "2":
        print(f"{mode} is not an option. Please enter 1 or 2.")

    else:
        if not os.path.exists("Results"):
            os.makedirs("Results")  # creates a Results folder if one does not exist

        tsp_filepath = input("Enter TSP filepath:\n> ")
        os.system("cls")
        print("Note: All routes end at the starting city.\n")
        if mode == "1":
            coords = parse_coords(tsp_filepath)  # get city coordinates from a single TSP file
            tsp_filename = tsp_filepath.split("/")[1]  # get TSP filename from filepath
            solve_TSP(coords, tsp_filename)  # find optimal route

        if mode == "2":
            results = []
            tsp_filenames = os.listdir(tsp_filepath)  # get all TSP filenames from a given directory
            tsp_filenames.sort(key=get_file_number)  # sort files in ascending order of number of cities in route

            # for each file: parse coordinates, find the optimal route, and record the results
            for tsp_filename in tsp_filenames:
                coords = parse_coords(f"{tsp_filepath}/{tsp_filename}")
                start_time = time()
                optimal_route, cost = solve_TSP(coords, tsp_filename)
                running_time = time() - start_time
                results.append([tsp_filename, optimal_route, cost, running_time])

            # create a DataFrame from the results and use pandas built-in function to write it to a CSV file in Results
            results_df = pd.DataFrame(results, columns=["File", "Optimal Route", "Cost", "Running Time (sec)"])
            write_to_csv(results_df)
    
        
