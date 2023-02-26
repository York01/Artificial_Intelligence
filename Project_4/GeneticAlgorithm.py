import random
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import pandas as pd
from time import time
import os

# global NetworkX and Matplotlib elements to enable animation
fig, ax = plt.subplots(1, 2, figsize=(12,6))  # require Axes object to add axes to Networkx graph
G = nx.Graph()  # create a Directed Graph object
node_sizes = [60] * 100
node_colors = ['Orange'] * 100
fig.suptitle(f"Random100 Genetic Algorithm Results", fontsize=16)

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
        self.elites = int(self.population_size/5)  # number of Chromosomes to ensure are selected
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


    def pop_by_value(self, value):
        '''
        Helper function to retrieve Chromosome while removing it, based on
        its value instead of index.
        '''
        for index, chromosome in enumerate(self.chromosomes):
            if value == chromosome:
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
        for i in range(int(self.population_size/2 - self.elites)):
            fitness_list = [chromosome.fitness for chromosome in self.chromosomes]
            choice = random.choices(self.chromosomes, weights=fitness_list, k=1)
            selection.append(self.pop_by_value(choice[0]))
        return selection

    def get_random_indices(self, chromosome):
        '''
        Helper function to get random indces based on length
        of chromosome.
        '''
        index1 = random.randint(0, len(chromosome))
        index2 = random.randint(0, len(chromosome))
        start = min(index1, index2)
        end = max(index1, index2)
        return (start, end)

    def crossover(self, selection):
        '''
        Crossover operator selects two Chromosomes to exchange data in the form
        of Gene subsequences. Each pair of parent Chromosomes creates a pair of
        child Chromosomes.
        '''
        random.shuffle(selection)  # randomize sequence of Genes
        children = []
        for i in range(0, len(selection), 2):
            parent1 = selection[i]
            parent2 = selection[i+1]
            genes1 = []
            genes2 = []

            # get random indices to select subsequence of Genes from
            start, end = self.get_random_indices(parent1)

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
        size = len(self.chromosomes[0])  # how many Genes to account for
        for chromosome in children:
            # determine whether to mutate the current Chromosome
            if (random.random() <= self.mutation_rate):
                genes = []

                # get random indices to select subsequence of Genes from
                start, end = self.get_random_indices(chromosome)

                # reverse subsequence of Genes b/w start and end and add back
                # into the same indices to mutate Chromosome
                for i in range(0, start):
                    genes.append(chromosome[i])
                for i in reversed(range(start, end)):
                    genes.append(chromosome[i])
                for i in range(end, size):
                    genes.append(chromosome[i])
                for i in range(size):
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
        gene = Gene(label, coord)
        genes.append(gene)
    return genes

def genetic_algorithm(parameters, genes):
    '''
    Implement genetic algorithm 
    '''
    # unpack parameters to specify how to run genetic algorithm
    population_size = parameters[0]
    mutation_rate = parameters[1]
    generations = parameters[2]
    population = Population(population_size, mutation_rate, genes)

    results = []
    # run genetic algorithm for however many generations specified
    for i in range(1, generations+1):
        selection = population.selection()
        children = population.crossover(selection)
        children = population.mutation(children)
        population.chromosomes = selection + children
        population.evaluation()

        # save the optimal result for the first and every one fifth of
        # generations to return after finishing
        if (i%(generations/5) == 0) or i==1:
            genes = []
            for gene in population.optimal:
                genes.append(gene)
            results.append([i, genes, population.optimal.cost])

    # compile statistics across all generations, including overall best and worst
    best = min(population.best_list)
    worst = max(population.worst_list)
    stats = [population.mean_list,
             population.best_list,
             population.worst_list,
             best,
             worst]
    return results, stats

def write_to_csv(filename, results, columns, parameters=None, optional=''):
    '''
    Write results of closest-edge insertion heuristic to a CSV file.
    '''
    if parameters:
        genes = []
        for gene in results[1]:
            genes.append(gene.label)
        results = [parameters + [results[2], genes]]
    final_results = pd.DataFrame(results, columns=columns)
    with open(f'Results/{filename}{optional}.csv', 'w', newline='') as file:
        file.write(final_results.to_csv(index=False))

def plot_generations(results, stats, generations, subtitle):
    ax[0].clear()
    ax[0].plot(range(1, generations+1), stats[0], label="Mean Cost")
    ax[0].plot(range(1, generations+1), [results[2]]*generations, label="Optimal Cost")
    ax[0].fill_between(range(1, generations+1), 
                     stats[1],
                     stats[2],
                     alpha=0.3,
                     label="Best-Worst Cost")
    ax[0].legend()
    ax[0].title.set_text(f'Cost over Generations\n{subtitle}')
    ax[0].set_ylabel('Cost')
    ax[0].set_xlabel('Generation')

def plot_route(chromosome, filename):
    fig, ax = plt.subplots()
    ax = plt.gca()  # get current axes, return new axis if none
    G = nx.Graph()
    genes = chromosome[1]
    populate_G(G, genes)
    nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=node_sizes,
            node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, nx.get_node_attributes(G, 'pos'), font_size=6)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.title.set_text(f"{filename} Optimal Tour")
    ax.set_axis_on()
    fig.savefig(f"Results/{filename}_Optimal.png", bbox_inches="tight")

def populate_G(G, genes):
    '''
    Helper function to add Genes as Nodes and connections between
    Genes as Edges to NetworkX graph G.
    '''
    for gene in genes:
        G.add_node(gene.label, pos=gene.coords)
    length = len(genes)
    for i in range(length):
        if i == length-1:
            G.add_edge(genes[i].label, genes[0].label)
        else:
            G.add_edge(genes[i].label, genes[i+1].label)

def update(chromosome):
    '''
    Plot Nodes and Edges of Network based on their coordinates. Save PNG image
    under Results folder.
    '''
    G.clear()
    ax[1].clear()
    # add each Node to Graph
    genes = chromosome[1]
    populate_G(G, genes)

    nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=node_sizes,
            node_color=node_colors, ax=ax[1])
    nx.draw_networkx_labels(G, nx.get_node_attributes(G, 'pos'), font_size=6)
    ax[1].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax[1].title.set_text(f'Optimal Tour at Generation {chromosome[0]}')
    ax[1].set_axis_on()

def get_best_index(best_runs):
    '''
    Helper function to retrieve index of best run.
    '''
    best_index = None
    for index, best_of_run in enumerate(best_runs):
        if best_index:
            if best_of_run < best_runs[best_index]:
                best_index = index
        else:
            best_index = index
    return best_index

def get_worst(worst_runs):
    '''
    Helper function to retrieve worst run.
    '''
    worst_run = None
    for worst_of_run in worst_runs:
        if worst_run:
            if worst_of_run < worst_run:
                worst_runs = worst_of_run
        else:
            worst_run = worst_of_run
    return worst_run

if __name__ == '__main__':
    os.system('cls')
    tsp_filepath = input('Enter .tsp filename to solve:\n> ')
    filename = tsp_filepath.split('.')[0]
    os.system('cls')
    if not os.path.exists('Results'):
        os.makedirs('Results')  # create Results folder if one does not exist

    coords = parse_coords(tsp_filepath)  # parse TSP file for coordinates
    genes = create_genes(coords)

    # initialize parameters to specify how to run the genetic algorithm
    generations = 1000
    population_sizes = [100, 500]  # population sizes to vary
    mutation_rates = [.01, .1]  # mutation rates to vary
    num_of_runs = 20  # number of times to run a single dataset

    summary_staistics = []
    # compile statistics for all four datasets across multiple runs
    for index1, population_size in enumerate(population_sizes):
        for index2, mutation_rate in enumerate(mutation_rates):
            print(f'Combination {index1} + {index2}')
            best_runs, worst_runs, running_time = [], [], 0
            results_list, stats_list = [], []

            # run genetic algorithm based on num_of_runs, store results and stats
            # for later use in plotting and determining the overall optimal solution
            for i in range(1, num_of_runs+1):
                print(f'Run {i}')
                parameters = [population_size, mutation_rate, generations]
                start_time = time()
                results, stats = genetic_algorithm(parameters, genes)
                running_time += time() - start_time
                results_list.append(results)
                stats_list.append(stats)
                best_runs.append(stats[3])
                worst_runs.append(stats[4])
            os.system('cls')

            best_index = get_best_index(best_runs)  # retrieve index to best result and best cost
            best_result = results_list[best_index][-1]

            # compile statistics for a single dataset
            best = best_runs[best_index]
            worst = get_worst(worst_runs)  # only retrieve worst cost (no need for index)
            mean = sum(best_runs)/num_of_runs
            standard_deviation = sqrt(sum([(value-mean)**2 for value in best_runs])/num_of_runs)
            running_time = round(running_time/num_of_runs, 2)

            summary_staistics.append([population_size, mutation_rate, mean, best, worst, standard_deviation, running_time])

            # write optimal solution for single dataset to a .csv fil
            combo = f'_{index1}{index2}'
            columns = ['Generations', 'Population Size', 'Mutation Rate', 'Cost', 'Genes']
            csv_parameters = [generations, population_size, mutation_rate]
            write_to_csv(filename, best_result, columns, csv_parameters, optional=combo)

            # plot optimal solution as 1) a subplot in dynamic GUI for change in cost over generations
            # and 2) the optimal solution itself as a graph of nodes and edges between them
            subtitle = f'Population: {population_size} | Mutation Rate: {mutation_rate}'
            plot_generations(best_result, stats_list[best_index], generations, subtitle)
            plot_route(best_result, f'{filename}{combo}')

            # create animation of the optimal solution changing over subsequent generations
            anim = mpl.animation.FuncAnimation(fig, update, frames=results_list[best_index], interval=4000, repeat=True, save_count=11)
            gif_filepath = f'Results/{filename}{combo}.gif' 
            gif_writer = mpl.animation.PillowWriter(fps=2)
            anim.save(gif_filepath, writer=gif_writer)

    # save statistics for all four datasets to a .csv file
    columns = ['Population Size', 'Mutation Rate', 'Mean', 'Min', 'Max', 'Standard Deviation', 'Running Time (s)']
    write_to_csv(filename, summary_staistics, columns, optional='_Stats')