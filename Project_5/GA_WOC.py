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
    def __init__(self, label, coords):
        self.label = str(label)
        self.coords = coords

    def __str__(self):
        return f'Gene {self.label}: {self.coords}'

class Chromosome:
    '''
    Represent a sequence of Genes, including their cost (length of path between coordinates)
    and fitness (based on cost).
    '''
    def __init__(self, genes):
        self.genes = genes # list of Genes
        self.cost = self.find_cost()  # length of path between coordinates in self.genes
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
        length = len(self.genes)
        cost = 0
        for index in range(0, length):
            gene1 = self.genes[index]
            if index == length-1:
                gene2 = self.genes[0]
            else:
                gene2 = self.genes[index+1]
            x_coord1 = gene1.coords[0]
            y_coord1 = gene1.coords[1]
            x_coord2 = gene2.coords[0]
            y_coord2 = gene2.coords[1]
            cost += sqrt((x_coord2 - x_coord1)**2 + (y_coord2 - y_coord1)**2)
        return cost

class Population:
    '''
    Represent a collection of Chromosomes with a variable population size (number of
    Chromosomes in collection) and mutation rate (probability of a Chromosome changing).
    '''
    def __init__(self, population_size, mutation_rate, genes):
        self.population_size = population_size  # number of Chromosomes in Population
        self.mutation_rate = mutation_rate  # probability of a Chromosomes changing
        self.elites = 50  # number of Chromosomes to ensure are selected
        self.optimal = None  # Chromosome encoding the optimal solution
        self.chromosomes = []  # list of Chromosomes

        # store statistics for each genrations (mean, min, max)
        self.mean_list = []
        self.best_list = []
        self.worst_list = []

        # initialize Chromosomes to create Population
        for i in range(1, self.population_size+1):
            chromosome = Chromosome(random.sample(genes, len(genes)))
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
            child1 = Chromosome(genes1)
            child2 = Chromosome(genes2)
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

def parse_coords(tsp_file):
    '''
    Parse coordinates from TSP file and return them as tuple pairs.
    '''
    f = open(tsp_file, "r")
    lines = f.readlines()
    size = len(lines)
    lines = lines[7:size]  # skips first seven lines of metadata
    coords = []
    for line in lines:
        _, coord_x, coord_y = line.replace("\n", "").split(" ")
        coords.append((float(coord_x), float(coord_y)))
    return coords

def create_genes(coords):
    '''
    Create Genes from each set of coordinates, using their sequence
    in which they're accessed as their labels.
    '''
    genes = []
    for label, coord in enumerate(coords):
        gene = Gene(label+1, coord)
        genes.append(gene)
    return genes

def GA(args):
    '''
    Implements genetic algorithm 
    '''
    # unpack parameters to specify how to run genetic algorithm
    parameters, genes = args
    population_size = parameters[0]
    mutation_rate = parameters[1]
    generations_to_run = parameters[2]
    population = Population(population_size, mutation_rate, genes)

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
            genes = []
            for gene in population.optimal:
                genes.append(gene)
            generations.append([i, genes, population.optimal.cost])

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
    with open(f'Results2/{filename}{optional}.csv', 'w', newline='') as file:
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
    fig.savefig(f"Results2/{filename}_{optional}Generations.png", bbox_inches="tight")

def plot_route(genes, filename, optional=''):
    fig, ax = plt.subplots()
    ax = plt.gca()  # get current axes, return new axis if none
    G = nx.Graph()
    for gene in genes:
        G.add_node(gene.label, pos=gene.coords)
    length = len(genes)
    for i in range(length):
        if i == length-1:
            G.add_edge(genes[i].label, genes[0].label)
        else:
            G.add_edge(genes[i].label, genes[i+1].label)

    nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=node_sizes,
            node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, nx.get_node_attributes(G, 'pos'), font_size=6)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.title.set_text(f"{filename} Optimal Tour{optional}")
    ax.set_axis_on()
    fig.savefig(f"Results2/{filename}_Tour.png", bbox_inches="tight")

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
    length = len(chromosomes_list[0][0])
    agreement_matrix = np.zeros((length, length))  # extra row & column so labels can be used as indices
    for chromosomes in chromosomes_list:
        for chromosome in chromosomes:
            for i in range(length):
                index1 = int(chromosome[i].label)-1
                if i == length-1:
                    index2 = int(chromosome[0].label)-1
                else:
                    index2 = int(chromosome[i+1].label)-1
                agreement_matrix[index1, index2] += 1
    return agreement_matrix

def find_maximum(index, new_genes, length, cost_matrix):
    current_col = None
    max_cost = None
    if index == -1:
        row = new_genes[index]
        for col in range(length):
            if col not in new_genes:
                cost = cost_matrix[row, col]
                if max_cost:
                    if cost > max_cost:
                        current_col = col
                        max_cost = cost
                else:
                    current_col = col
                    max_cost = cost
        return current_col, max_cost
    else:
        col = new_genes[index]
        for row in range(length):
            if row not in new_genes:
                cost = cost_matrix[row, col]
                if max_cost:
                    if cost > max_cost:
                        current_row = row
                        max_cost = cost
                else:
                    current_row = row
                    max_cost = cost
        return current_row, max_cost

def find_minimum(index, new_genes, length, cost_matrix):
    current_col = None
    min_cost = None
    if index == -1:
        row = new_genes[index]
        for col in range(length):
            if col not in new_genes:
                cost = cost_matrix[row, col]
                if min_cost:
                    if cost < min_cost:
                        current_col = col
                        min_cost = cost
                else:
                    current_col = col
                    min_cost = cost
        return current_col, min_cost
    else:
        col = new_genes[index]
        for row in range(length):
            if row not in new_genes:
                cost = cost_matrix[row, col]
                if min_cost:
                    if cost < min_cost:
                        current_row = row
                        min_cost = cost
                else:
                    current_row = row
                    min_cost = cost
        return current_row, min_cost

def combine_solutions(genes, agreement_matrix, cost_matrix):
    length = len(genes)
    new_genes = []
    i = np.argmax(agreement_matrix)  # returns a flattened array & finds index based on that
    row = floor(i/length)
    col = i%length
    new_genes = [row, col]  # row is the city travelled from, col is the city traveled to
    new_length = len(new_genes)
    last_index_inserted = -1
    while new_length < length:

        prev_col, prev_agreement = find_maximum(0, new_genes, length, agreement_matrix)
        next_col, next_agreement = find_maximum(-1, new_genes, length, agreement_matrix)

        if prev_agreement > next_agreement:
            new_genes.insert(0, prev_col)
            last_index_inserted = 0
        elif prev_agreement < next_agreement:
            new_genes.append(next_col)
            last_index_inserted = -1
        else:
            new_col, new_cost = find_minimum(last_index_inserted, new_genes, length, cost_matrix)
            if last_index_inserted == -1:
                new_genes.append(new_col)
                last_index_inserted = 0
            else:
                new_genes.insert(0, new_col)
                last_index_inserted = -1
        new_length += 1
    optimal = Chromosome([genes[new_gene_index] for new_gene_index in new_genes])
    return optimal

def find_cost_matrix(genes):
    '''
    Find cost based on distance b/w Genes. Used to find
    closest neighbor for WOC when a node is not available
    in the agreement matrix.
    cost = 0
    cost += sqrt((x2 - x1)^2 + (y2 - y1)^2)
    '''
    length = len(genes)
    cost_matrix = np.zeros((length, length))
    for index1 in range(length):
        for index2 in range(length):
            if index1 != index2:
                x_coord1 = genes[index1].coords[0]
                y_coord1 = genes[index1].coords[1]
                x_coord2 = genes[index2].coords[0]
                y_coord2 = genes[index2].coords[1]
                cost_matrix[index1, index2] = sqrt((x_coord2 - x_coord1)**2 + (y_coord2 - y_coord1)**2)
    return cost_matrix

def solve_TSP(coords, tsp_filename):
    filename = tsp_filename.split(".")[0]
    genes = create_genes(coords)

    # set parameters for NetworkX graphs
    global node_sizes
    node_sizes = [60] * len(genes)
    global node_colors
    node_colors = ['Orange'] * len(genes)

    # initialize parameters to specify how to run the genetic algorithm
    files = ['Random11', 'Random22', 'Random44', 'Random77', 'Random97', 'Random222']
    generations_list = [20, 100, 1000, 1000, 1500, 2000]
    generations_to_run = 100  # default
    for index, file in enumerate(files):
        if filename == file:
            generations_to_run = generations_list[index]

    population_size = 500  # population sizes to vary
    mutation_rate = .1  # mutation rates to vary
    num_of_runs = 10  # number of times to run a single dataset
    parameters = [population_size, mutation_rate, generations_to_run]

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
    average_runtime_GA = round(total_runtime_GA/num_of_runs, 2)

    # run WOC for different percentages of chromosomes from each population
    # of the ten populations used (from 1-20%, incrementing by 1% each time).
    cost_list_WOC, total_runtime_WOC = [], 0
    best_result_WOC = None
    elites_used = None
    for i in range(5, 105, 5):
        chromosomes_list = []
        for results in results_list:
            chromosomes_list.append(results[:i])

        start_time = time()
        agreement_matrix = create_agreement_matrix(chromosomes_list, i)
        cost_matrix = find_cost_matrix(genes)
        result_WOC = combine_solutions(genes, agreement_matrix, cost_matrix)
        running_time = time() - start_time
        print(f'Running Time for WOC: {running_time}')

        cost_list_WOC.append(result_WOC.cost)
        total_runtime_WOC += running_time
        if best_result_WOC:
            if result_WOC.cost < best_result_WOC.cost:
                best_result_WOC = result_WOC
                elites_used = i
        else:
            best_result_WOC = result_WOC
            elites_used = i

    # compile WOC statistics for a single dataset
    best_WOC = round(best_result_WOC.cost, 2)
    mean_WOC = round(mean(cost_list_WOC), 2)
    standard_deviation_WOC = round(sqrt(sum([(value-mean_WOC)**2 for value in cost_list_WOC])/20), 2)
    average_runtime_WOC = round(total_runtime_WOC, 2)
    percentage_difference = 1 - best_WOC / best_GA
    total_runtime_combined = total_runtime_GA + total_runtime_WOC
    # plot tour of best result for WOC
    plot_route(best_result_WOC.genes, f'{filename}_WOC_Top{elites_used}')

    # plot tour of best result for GA over successive generations
    for generation in generations_GA:
        plot_route(generation[1], f'{filename}_GA_{generation[0]}', f'\nGeneration {generation[0]}')

    # plot tour and line plot of best result for GA 
    plot_route(best_result_GA.genes, f'{filename}_GA')
    plot_generations(generations_GA[-1], stats_GA, generations_to_run, f'{filename}_GA')

    # write statistics for GA and WOC for a single dataset to a CSV file
    columns = ['Generations',
               'Best GA',
               'Avg. GA',
               'Std Dev GA',
               'Avg. Runtime GA (s)',
               'Total Runtime GA (s)',
               'Best WOC',
               'Avg. WOC',
               'Std Dev WOC',
               'Avg. Runtime WOC (ms)',
               'Total Runtime WOC (s)',
               'Elites Used',
               '% Difference',
               'Total Runtime GA+WOC']

    csv_parameters = [generations_to_run,
                      best_GA,
                      mean_GA,
                      standard_deviation_GA,
                      average_runtime_GA,
                      total_runtime_GA,
                      best_WOC,
                      mean_WOC,
                      standard_deviation_WOC,
                      average_runtime_WOC,
                      total_runtime_WOC,
                      elites_used,
                      percentage_difference,
                      total_runtime_combined]

    write_to_csv(filename, columns, csv_parameters)

if __name__ == '__main__':
    os.system('cls')
    mode = input("Enter 1 if you want to solve for a single TSP file and plot the route.\n"
                 "Enter 2 if you want to solve for all TSP files in a given"
                 " directory, plot each route, and write the results to a CSV file.\n> ")

    if mode != '1' and mode != '2':
        print(f'{mode} is not an option. Please enter 1 or 2.')
        exit()

    if not os.path.exists('Results2'):
            os.makedirs('Results2')  # create Results folder if one does not exist

    tsp_filepath = input("Enter TSP filepath:\n> ")
    os.system('cls')

    if mode == '1':
        coords = parse_coords(tsp_filepath)  # get city coordinates from a single TSP file
        tsp_filename = tsp_filepath.split("/")[1]  # get TSP filename from filepath
        solve_TSP(coords, tsp_filename)  # find optimal route

    elif mode == '2':
        results = []
        tsp_filenames = os.listdir(tsp_filepath)  # get all TSP filenames from a given directory
        tsp_filenames.sort(key=get_file_number)  # sort files in ascending order of number of cities in route

        # for each file: parse coordinates, find the optimal route, and record the results
        for tsp_filename in tsp_filenames:
            coords = parse_coords(f"{tsp_filepath}/{tsp_filename}")
            solve_TSP(coords, tsp_filename)