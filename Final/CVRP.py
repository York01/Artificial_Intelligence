# graphing libraries for plotting graphs
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

# advanced data structure libraries for constructing matrices & creating dataframes
import pandas as pd
import numpy as np

# time library for access to time function to track running time  
from time import time

# os library for access functions related to directories, pausing execution, and clearing the screen
import os

# math & statistics libraries for access to math functions and constants
from math import sqrt, inf, floor
from statistics import mean

# random library for access to functions for selecting random integers, sampling lists randomly, etc.
import random

# datetime library for access to datetime function to create a unique random seed
from datetime import datetime
random.seed(datetime.now())


class Gene:
    '''
    Represent a point in 2D-space by its name ("label") and coordinates ("coords").
    '''
    def __init__(self, label, coords, demand):
        self.label = label
        self.coords = coords
        self.demand = demand

    def __str__(self):
        return f'Gene {self.label}: {self.coords} {self.demand}'

class Chromosome:
    '''
    Represent a sequence of Genes, including their cost (length of path between coordinates)
    and fitness (based on cost).
    '''
    def __init__(self, genes, depot, capacity, routes=False):
        self.genes = genes # list of Genes
        self.depot = depot
        self.capacity = capacity
        if routes:
            self.routes = routes
        else:
            self.routes = self.find_routes()  # each route based capacity per vehicle
        self.cost = self.find_cost()  # length of path for each route in self.routes
        self.fitness = 1/self.cost # lower cost => higher fitness

    def __lt__(self, other):
        return True

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, index):
        return self.genes[index]

    def __setitem__(self, index, value):
        self.genes[index] = value

    def find_cost(self):
        '''
        Find self.cost based on the distance b/w Genes' coordinates, including the last
        and first Gene.
        cost += sqrt((x2 - x1)^2 + (y2 - y1)^2)
        '''
        cost = 0
        for route in self.routes:
            length = len(route)
            for index in range(0, length):
                gene1 = route[index]
                if index == length-1:
                    gene2 = route[0]
                else:
                    gene2 = route[index+1]
                x_coord1 = gene1.coords[0]
                y_coord1 = gene1.coords[1]
                x_coord2 = gene2.coords[0]
                y_coord2 = gene2.coords[1]
                cost += sqrt((x_coord2 - x_coord1)**2 + (y_coord2 - y_coord1)**2)
        return cost

    def find_routes(self):
        '''
        Find routes for the VRP based on vehicle capacity. Each route
        includes as many customers as capacity allows. When capacity
        is full, a new route is built and customersa added to it.
        '''
        demand = 0
        routes = []
        route = [self.depot]
        length = len(self.genes)
        for i, gene in enumerate(self.genes):
            demand += gene.demand
            if self.capacity - demand >= 0:
                route.append(gene)
            else:
                routes.append(route)
                demand = gene.demand
                route = [self.depot]
                route.append(gene)
            if i == length - 1:
                routes.append(route)

        return routes

class Population:
    '''
    Represent a collection of Chromosomes with a variable population size (number of
    Chromosomes in collection) and mutation rate (probability of a Chromosome changing).
    '''
    def __init__(self, population_size, mutation_rate, genes, depot, capacity):
        self.population_size = population_size  # number of Chromosomes in Population
        self.mutation_rate = mutation_rate  # probability of a Chromosomes changing
        self.elites = 20  # number of Chromosomes to ensure are selected
        self.optimal = None  # Chromosome encoding the optimal solution
        self.chromosomes = []  # list of Chromosomes
        self.depot = depot
        self.capacity = capacity

        # store statistics for each genrations (mean, min, max)
        self.mean_list = []
        self.best_list = []
        self.worst_list = []

        # initialize Chromosomes to create Population
        length = len(genes)
        for i in range(1, self.population_size+1):
            chromosome = Chromosome(random.sample(genes, length), self.depot, self.capacity)
            self.chromosomes.append(chromosome)

        self.chromosome_length = len(self.chromosomes)

    def pop_by_value(self, value, fitness_list):
        '''
        Helper function to retrieve Chromosome while removing it, based on
        its value instead of index.
        '''
        for index, chromosome in enumerate(self.chromosomes):
            if value == chromosome:
                del fitness_list[index]
                return self.chromosomes.pop(index)
        return None

    def selection(self):
        '''
        Select which Chromosomes reproduce. Ranks Chromosomes from most to least fit and
        automatically selects the most fit based on self.elites. The remainder are selected
        based on probability associated with their fitness.
        '''
        # sort Chromosomes from low to high cost (same difference as fitness)
        to_sort = zip([chromosome.cost for chromosome in self.chromosomes], self.chromosomes)
        self.chromosomes = [x for y, x in sorted(to_sort)]

        # select number of Chromosomes based on self.elites (number to ensure reproduce)
        selection = []
        for i in range(self.elites):
            selection.append(self.chromosomes.pop(0))

        # select remainder based on probability, with a lower cost leading to a higher fitness
        # and subsequently a greater probability
        fitness_list = [chromosome.fitness for chromosome in self.chromosomes]
        for i in range(int(self.population_size/2 - self.elites)):
            choice = random.choices(self.chromosomes, weights=fitness_list, k=1)
            selection.append(self.pop_by_value(choice[0], fitness_list))
        return selection, selection[:self.elites]

    def get_random_indices(self, length):
        '''
        Helper function to get random indces based on length
        of chromosome.
        '''
        index1, index2 = 0, 0
        while index1 == index2:
            index1 = random.randint(0, length)
            index2 = random.randint(0, length)
        start = min(index1, index2)
        end = max(index1, index2)
        return (start, end)

    def crossover(self, selection):
        '''
        Crossover operator selects two Chromosomes to exchange data in the form
        of Gene subsequences. Each pair of parent Chromosomes creates a pair of
        child Chromosomes.
        '''
        length = self.chromosome_length - self.elites
        children = []
        while(len(children) < length):
            index1, index2 = self.get_random_indices(len(selection)-1)
            parent1 = selection[index1]
            parent2 = selection[index2]
            genes1 = []
            genes2 = []

            # get random indices to select subsequence of Genes from
            start, end = self.get_random_indices(len(parent1))

            # append subsequence of Genes b/w start and end indices to genes1 for first child
            for i in range (start, end):
                genes1.append(parent1[i])
            # append subsrquence of Genes up to start index to genes2 for second child
            for i in range(start):
                genes2.append(parent1[i])

            # append Genes from parent2 not yet in genes1, else append to genes2
            for gene in parent2:
                if gene not in genes1:
                    genes1.append(gene)
                else:
                    genes2.append(gene)
            # append subsequence of Genes from end index up to the last Gene in parent1 to genes2
            for i in range(end, len(parent1)):
                genes2.append(parent1[i])

            # initialize child Chromosomes and append to children
            child1 = Chromosome(genes1, self.depot, self.capacity)
            child2 = Chromosome(genes2, self.depot, self.capacity)
            children.append(child1)
            children.append(child2)
        return children

    def mutation(self, children):
        '''
        Mutation operator to change Chromosomes based on probability (specified
        by mutation rate) in some random manner based on Reverse Sequence Mutation. 
        '''
        length = len(self.chromosomes[0])  # how many Genes to account for
        for chromosome in children:
            # determine whether to mutate the current Chromosome
            if (random.random() <= self.mutation_rate):
                genes = []

                # get random indices to select subsequence of Genes from
                start, end = self.get_random_indices(length)

                # reverse subsequence of Genes b/w start and end and add back
                # into the same indices to mutate Chromosome
                for i in range(0, start):
                    genes.append(chromosome[i])
                for i in reversed(range(start, end)):
                    genes.append(chromosome[i])
                for i in range(end, length):
                    genes.append(chromosome[i])
                for i in range(length):
                    chromosome[i] = genes[i]
                chromosome.cost = chromosome.find_cost()
        return children

    def evaluation(self):
        '''
        Find optimal Chromosome (based on lowest cost) and store
        mean, min, and max for compiling statistics.
        '''
        costs = []
        for chromosome in self.chromosomes:
            cost = chromosome.cost
            if self.optimal:
                if(cost < self.optimal.cost):
                        self.optimal = chromosome
            else:
                self.optimal = chromosome
            costs.append(cost)
        self.mean_list.append(sum(costs) / self.population_size)
        self.best_list.append(min(costs))
        self.worst_list.append(max(costs))

def parse_coords(vrp_file):
    '''
    Parse coordinates from CVRP file and return them as tuple pairs.
    '''
    f = open(vrp_file, "r")
    lines = f.readlines()

    dimensions = int(lines[3].split(':')[1].strip())
    capacity = int(lines[5].split(':')[1].strip())

    start_node_coords = 7
    end_node_coords = start_node_coords + dimensions
    node_coords = lines[start_node_coords:end_node_coords]  # skips first seven lines of metadata

    start_demand = end_node_coords + 1
    end_demand = start_demand + dimensions
    demand = lines[start_demand:end_demand]  # skips first seven lines of metadata

    depot_index = int(lines[-3]) - 1

    coords = []
    for line in node_coords:
        _, coord_x, coord_y = line.replace("\n", "").split(" ")
        coord_x, coord_y = int(coord_x), int(coord_y)
        coords.append((coord_x, coord_y))

    demands = []
    for line in demand:
        _, demand = line.replace("\n", "").split(" ")
        demand = int(demand)
        demands.append(demand)

    parameters = {"depot_index": depot_index, "capacity": capacity, "coords": coords, "demands": demands}
    return parameters

def create_genes(coords, demands, depot_index):
    '''
    Create Genes from each set of coordinates, using their sequence
    in which they're accessed as their labels.
    '''
    genes = []
    depot_coords = coords.pop(depot_index)
    depot_demand = demands.pop(depot_index)
    depot = Gene(depot_index+1, depot_coords, depot_demand)
    for i in range(len(coords)):
        gene = Gene(i+1, coords[i], demands[i])
        genes.append(gene)
    return genes, depot

def GA(args):
    '''
    Implements genetic algorithm 
    '''
    # unpack parameters to specify how to run genetic algorithm
    parameters, genes = args
    population_size = parameters[0]
    mutation_rate = parameters[1]
    generations_to_run = parameters[2]
    depot = parameters[3]
    capacity = parameters[4]
    population = Population(population_size, mutation_rate, genes, depot, capacity)

    generations = []
    # run genetic algorithm for however many generations specified
    for i in range(1, generations_to_run+1):
        selection, elites = population.selection()
        children = population.crossover(selection)
        children = population.mutation(children)
        population.chromosomes = elites + children
        population.evaluation()
        # save the optimal result for the first and every one fifth of
        # generations to return after finishing
        if (i%(generations_to_run/5) == 0) or i==1:
            route_list = []
            routes_list = []
            for route in population.optimal.routes:
                for gene in route:
                    route_list.append(gene)
                routes_list.append(route_list)
                route_list = []
            generations.append([i, routes_list, population.optimal.cost])

    # compile statistics across all generations, including overall best and worst
    best = min(population.best_list)
    worst = max(population.worst_list)
    stats = [population.mean_list,
             population.best_list,
             population.worst_list,
             best,
             worst]
    to_sort = zip([chromosome.cost for chromosome in population.chromosomes], population.chromosomes)
    results = [x for y, x in sorted(to_sort)]
    return results, stats, generations

def write_to_csv(filename, columns, parameters=None, optional=''):
    '''
    Write results of closest-edge insertion heuristic to a CSV file.
    '''
    final_results = pd.DataFrame([parameters], columns=columns)
    with open(f'Results/{filename}{optional}.csv', 'w', newline='') as file:
        file.write(final_results.to_csv(index=False))

def plot_generations(results, stats, generations, filename, optional=''):
    fig, ax = plt.subplots()
    ax = plt.gca()  # get current axes, return new axis if none
    ax.plot(range(1, generations+1), stats[0], label="Mean Cost")
    ax.plot(range(1, generations+1), [results[2]]*generations, label="Optimal Cost")
    ax.fill_between(range(1, generations+1), 
                     stats[1],
                     stats[2],
                     alpha=0.3,
                     label="Best-Worst Cost")
    ax.legend()
    ax.title.set_text(f'Cost over Generations for Optimal Chromosome')
    ax.set_ylabel('Cost')
    ax.set_xlabel('Generation')
    fig.savefig(f"Results/{filename}_{optional}Generations.png", bbox_inches="tight")

def plot_routes(chromosome, filename, capacity, depot, type, optional='', mode=1):
    fig, ax = plt.subplots()
    ax = plt.gca()  # get current axes, return new axis if none
    G = nx.Graph()

    # genes and depot are added to graph
    G.add_node(depot.label, pos=depot.coords, color='yellow')
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', 'limegreen', 'tomato',
              'lightpink', 'peru']
    routes = None
    if mode == 1:
        routes = chromosome.routes
    else:
        routes = chromosome

    j = 0
    for route in routes:
        for n, gene in enumerate(route[1:]):  # first gene is always depot
            G.add_node(gene.label, pos=gene.coords, color='white')

        length = len(route)
        for i in range(length):
            if i == length-1:
                G.add_edge(route[i].label, route[0].label, color=colors[j])
            else:
                G.add_edge(route[i].label, route[i+1].label, color=colors[j])
        j += 1

    size = [1 for node in G.nodes]
    node_sizes = [60] * len(size)
    node_colors = [node[1]['color'] for node in G.nodes(data=True)]
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=node_sizes,
            node_color=node_colors, edge_color=edge_colors, ax=ax)
    nx.draw_networkx_labels(G, nx.get_node_attributes(G, 'pos'), font_size=6)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.title.set_text(f"{filename} {type} Tour - {optional} Generations")
    ax.set_axis_on()
    plt.show()
    png_filename = None
    if mode == 1:
        png_filename = f"Results/{filename}_{type}_Tour.png"
    else:
        png_filename = f"Results/{filename}_{type}_{optional}_Tour.png"
    fig.savefig(png_filename, bbox_inches="tight")

def get_best_result(args):
    '''
    Helper function to retrieve index of best run.
    '''
    best_result = None
    for lists in args:
        for result, stats, generations in args:
            if best_result:
                if result.cost < best_result.cost:
                    best_result = result
                    best_stats = stats
                    best_generations = generations
            else:
                best_result = result
                best_stats = stats
                best_generations = generations
    return best_result, best_stats, best_generations

def create_agreement_matrix(chromosomes_list, elites_used):
    '''
    Create 2D matrix with gene labels on the x- and y-axes.
    The intersection of the row and column tells how many
    individuals ordered their chromosome to visit the city on the
    x-axis from the city on the y-axis.
    '''
    length = len(chromosomes_list[0][0])+1
    agreement_matrix = np.zeros((length, length))  # extra row & column so labels can be used as indices
    i = 0
    for chromosomes in chromosomes_list:
        for chromosome in chromosomes:
            for route in chromosome.routes:
                route_length = len(route)
                for i in range(0, route_length):
                    index1 = route[i].label - 1
                    if i == route_length-1:
                        index2 = route[0].label - 1
                    else:
                        index2 = route[i+1].label - 1
                    agreement_matrix[index1, index2] += 1
    return agreement_matrix

def find_maximum(index, new_genes, genes, demand, capacity, agreement_matrix):
    current_col = None
    max_cost = None
    length = len(genes)
    row = new_genes[index]
    for col in range(length):
        if (col not in new_genes) and ((capacity - (demand + genes[col].demand) >= 0)):
            cost = agreement_matrix[row, col]
            if max_cost:
                if cost > max_cost:
                    current_col = col
                    max_cost = cost
            else:
                current_col = col
                max_cost = cost
    if current_col:
        return current_col, max_cost
    else:
        return 0, 0

def combine_solutions(genes, agreement_matrix, depot, capacity):
    '''
    Combine aggregated solutions based on agreement_matrix to build a new solution
    using WoAC.
    '''
    depot_index = depot.label-1

    length = len(genes)
    new_genes = []
    max_index = np.argmax(agreement_matrix[0,:])  # returns a flattened array & finds index based on that
    previous_max_indices = [max_index]

    new_genes = [0, max_index]  # row is the city travelled from, col is the city traveled to
    new_length = len(new_genes)

    demand = genes[max_index].demand
    route = [depot, genes[max_index]]
    routes = []
    while new_length < length:

        next_col, _ = find_maximum(new_genes, genes, demand, capacity, agreement_matrix)
        demand += genes[next_col].demand
        if next_col != depot_index:
            new_genes.append(next_col)
            route.append(genes[next_col])
        else:
            routes.append(route)
            new_max_index = None
            max_agreement = None
            for index, agreement in enumerate(agreement_matrix[0, :]):
                if (index not in previous_max_indices) and (index not in new_genes):
                    if new_max_index:
                        if agreement < max_agreement:
                            new_max_index = index
                            max_agreement = agreement
                    else:
                        new_max_index = index
                        max_agreement = agreement
            previous_max_indices.append(new_max_index)
            demand = genes[new_max_index].demand
            route = [depot, genes[new_max_index]]
            new_genes.append(new_max_index)
        new_length += 1
    if new_length == length:
        routes.append(route)
    optimal = Chromosome([genes[new_gene_index] for new_gene_index in new_genes], depot, capacity, routes)
    return optimal

def solve_CVRP(parameters, cvrp_filename):
    '''
    Main function for solving CVRP. Initializes values based on file
    parameters, runs the GA, WoAC, and compiles results and statistics.
    '''
    filename = cvrp_filename.split(".")[0]
    depot_index, capacity, coords, demands = parameters.values()
    genes, depot = create_genes(coords, demands, depot_index)

    generations_to_run = 200
    population_size = 100  # population sizes to vary
    mutation_rate = .1  # mutation rates to vary
    num_of_runs = 10  # number of times to run a single dataset
    parameters = [population_size, mutation_rate, generations_to_run, depot, capacity]

    # compile statistics for all four datasets across multiple runs
    best_runs, worst_runs, total_runtime_GA = [], [], 0
    results_list, stats_list, generations_list = [], [], []

    # run  based on num_of_runs, store results (Chromosomes),
    # stats (statistics across all generations of the population), and
    # generations (sequence of routes and costs across generations)
    # for later use in plotting and determining the overall optimal solution
    for i in range(1, num_of_runs+1):
        print(f'Run {i}')
        start_time = time()
        results, stats, generations = GA((parameters, genes))
        total_runtime_GA += time() - start_time
        print(f'Finished Running in: {time()- start_time} s')
        results_list.append(results)
        stats_list.append(stats)
        generations_list.append(generations)
        best_runs.append(stats[3])
        worst_runs.append(stats[4])
    os.system('cls')

    args_tuple = zip(results_list, stats_list, generations_list)
    args = [[results[0], stats, generations] for results, stats, generations in args_tuple]
    best_result_GA, stats_GA, generations_GA = get_best_result(args)

    # compile GA statistics for a single dataset
    best_GA = round(best_result_GA.cost, 2)
    mean_GA = round(mean(best_runs), 2)
    standard_deviation_GA = round(sqrt(sum([(value-mean_GA)**2 for value in best_runs])/num_of_runs), 2)
    vehicles_GA = len(best_result_GA.routes)
    average_runtime_GA = round(total_runtime_GA/num_of_runs, 2)

    # run WoAC for different percentages of chromosomes from each population
    # of the ten populations used (from 1-20%, incrementing by 1% each time).
    cost_list_WoAC, total_runtime_WoAC = [], 0
    best_result_WoAC = None
    elites_used = None
    for gene in genes:
        if gene.label >= depot.label:
            gene.label += 1
    for i in range(5, 25, 5):
        chromosomes_list = []
        for results in results_list:
            chromosomes_list.append(results[:i])  # CHANGE BACK TO [:i]

        start_time = time()
        agreement_matrix = create_agreement_matrix(chromosomes_list, i)
        new_genes = [depot] + [gene for gene in genes]
        result_WoAC = combine_solutions(new_genes, agreement_matrix, depot, capacity)
        running_time = time() - start_time
        print(f'Running Time for WoAC: {running_time}')

        cost_list_WoAC.append(result_WoAC.cost)
        total_runtime_WoAC += running_time
        if best_result_WoAC:
            if result_WoAC.cost < best_result_WoAC.cost:
                best_result_WoAC = result_WoAC
                elites_used = i
        else:
            best_result_WoAC = result_WoAC
            elites_used = i

    # compile WoAC statistics for a single dataset
    best_WoAC = round(best_result_WoAC.cost, 2)
    mean_WoAC = round(mean(cost_list_WoAC), 2)
    standard_deviation_WoAC = round(sqrt(sum([(value-mean_WoAC)**2 for value in cost_list_WoAC])/20), 2)
    vehicles_WoAC = len(best_result_WoAC.routes)
    average_runtime_WoAC = round(total_runtime_WoAC/4, 2)
    percentage_difference = 1 - best_WoAC / best_GA
    total_runtime_combined = total_runtime_GA + total_runtime_WoAC
    # plot tour of best result for WoAC
    plot_routes(best_result_WoAC, f'{filename}_Top{elites_used}', capacity, depot, 'WoAC', f'{generations_to_run}')

    # plot tour of best result for GA over successive generations
    for generation in generations_GA:
        plot_routes(generation[1], f'{filename}', capacity, depot, 'GA', generation[0], mode=2)

    # plot tour and line plot of best result for GA
    best_result_GA.genes.insert(0, depot)
    plot_routes(best_result_GA, f'{filename}', capacity, depot, 'GA', f'{generations_to_run}')
    plot_generations(generations_GA[-1], stats_GA, generations_to_run, f'{filename}_GA')

    # write statistics for GA and WoAC for a single dataset to a CSV file
    columns = ['Generations',
               'Best GA',
               'Avg. GA',
               'Std Dev GA',
               'Vehicles GA',
               'Avg. Runtime GA (s)',
               'Total Runtime GA (s)',
               'Best WoAC',
               'Avg. WoAC',
               'Std Dev WoAC',
               'Vehicles WoAC',
               'Avg. Runtime WoAC (ms)',
               'Total Runtime WoAC (s)',
               'Elites Used',
               '% Difference',
               'Total Runtime GA+WoAC']

    csv_parameters = [generations_to_run,
                      best_GA,
                      mean_GA,
                      standard_deviation_GA,
                      vehicles_GA,
                      average_runtime_GA,
                      total_runtime_GA,
                      best_WoAC,
                      mean_WoAC,
                      standard_deviation_WoAC,
                      vehicles_WoAC,
                      average_runtime_WoAC,
                      total_runtime_WoAC,
                      elites_used,
                      percentage_difference,
                      total_runtime_combined]

    write_to_csv(filename, columns, csv_parameters)

if __name__ == '__main__':
    os.system('cls')
    mode = input("Enter 1 if you want to solve for a single CVRP file and plot the route.\n"
                 "Enter 2 if you want to solve for all CVRP files in a given"
                 " directory, plot each route, and write the results to a CSV file.\n> ")

    if mode != '1' and mode != '2':
        print(f'{mode} is not an option. Please enter 1 or 2.')
        exit()

    if not os.path.exists('Results'):
            os.makedirs('Results')  # create Results folder if one does not exist

    cvrp_filepath = input("Enter CVRP filepath:\n> ")
    os.system('cls')

    if mode == '1':
        parameters = parse_coords(cvrp_filepath)  # get city coordinates from a single CVRP file
        cvrp_filename = cvrp_filepath.split("/")[1]  # get CVRP filename from filepath
        solve_CVRP(parameters, cvrp_filename)  # find optimal route

    elif mode == '2':
        results = []
        cvrp_filenames = os.listdir(cvrp_filepath)  # get all CVRP filenames from a given directory
        cvrp_filenames.sort(key=get_file_number)  # sort files in ascending order of number of cities in route

        # for each file: parse coordinates, find the optimal route, and record the results
        for cvrp_filename in cvrp_filenames:
            coords = parse_coords(f"{cvrp_filepath}/{cvrp_filename}")
            solve_CVRP(coords, cvrp_filename)