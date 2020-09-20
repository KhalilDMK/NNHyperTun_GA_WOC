from random import randint
import random
import collections
import math
from Code.model import cost_of_solution


# Method to create a random solution

def create_random_solution():
    nb_neurons = [8, 16, 32, 64]
    nb_layers = [2, 4, 5, 6, 8]
    batch_size = [8, 16, 32, 64, 128]
    optimizer = ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']
    activation = ['relu', 'elu', 'tanh', 'sigmoid']

    solution = []
    for hyperparameter in ['nb_neurons', 'nb_layers', 'batch_size', 'optimizer', 'activation']:
        solution.append(random.choice(eval(hyperparameter)))
    return solution


# Method to create initial population

def create_initial_population(population_size):
    initial_population = [create_random_solution() for i in range(population_size)]
    return initial_population

# Method to create a dictionary of chromosomes with their fitness

def create_dictionary_with_percentages_of_cumulated_fitness(population, epochs, x_train, y_train, x_test, y_test):
    fitness_dictionary = dict((str(element), 0) for element in population)
    solution_dictionary = dict((str(element), 0) for element in population)
    accuracy_dictionnary = dict((str(element), 0) for element in population)

    for solution in population:
        accuracy = cost_of_solution(solution, epochs, x_train, y_train, x_test, y_test)
        if math.isnan(accuracy):
            accuracy = float('inf')
        print('current working solution is ', solution, 'error is ', accuracy)

        fitness_dictionary[str(solution)] = 1 / (accuracy + 1)
        accuracy_dictionnary[str(solution)] = accuracy
        solution_dictionary[str(solution)] = solution
    sum_of_fitnesses = sum(fitness_dictionary.values())
    for solution in population:
        fitness_dictionary[str(solution)] = fitness_dictionary[str(solution)] / sum_of_fitnesses
    return fitness_dictionary, solution_dictionary, accuracy_dictionnary


# Method for random wheel selection

def random_wheel_selection(fitness_dictionary, solution_dictionary):
    key = random.choice(list(fitness_dictionary))
    return solution_dictionary[key]


# Method for crossover using a random cut point

def crossover_with_random_cutpoint(parent_1, parent_2):
    cut_point = randint(0, len(parent_1) - 2)
    crossover_child = parent_1[0:cut_point] + parent_2[cut_point:len(parent_2)]
    return crossover_child


# Method for crossover using a section in the middle with random length

def crossover_with_section_in_middle(parent_1, parent_2):
    cut_point_1 = randint(0, len(parent_1) - 2)
    cut_point_2 = cut_point_1 + randint(1, len(parent_1) - cut_point_1 - 1)
    crossover_child = parent_2[0:cut_point_1] + parent_1[cut_point_1:cut_point_2 + 1] + parent_2[
                                                                                        cut_point_2 + 1:len(parent_2)]
    return crossover_child


# Method to generate the crossover solutions

def generate_crossover_solutions(population, percentage_crossover, crossover_method, epochs, x_train, y_train, x_test,
                                 y_test, fitness_dictionary, solution_dictionary):
    Number_of_crossover_solutions = int(percentage_crossover * len(population))
    crossover_solutions = []
    child = []
    for i in range(Number_of_crossover_solutions):
        parent_1, parent_2 = 0, 0

        counter = 0
        while (child in crossover_solutions or child == []) and counter < 100:
            counter += 1
            while type(parent_1) != list or type(parent_2) != list or parent_1 == parent_2:
                parent_1 = random_wheel_selection(fitness_dictionary, solution_dictionary)
                parent_2 = random_wheel_selection(fitness_dictionary, solution_dictionary)
            if crossover_method == 'Random cutpoint':
                child = crossover_with_random_cutpoint(parent_1, parent_2)
            elif crossover_method == 'Random section in middle':
                child = crossover_with_section_in_middle(parent_1, parent_2)
        if counter == 100:
            child = create_random_solution()
            print('Random solution created because of loop')

        crossover_solutions.append(child)

    return crossover_solutions


# Method to generate mutated solutions

def generate_mutated_solutions(population, percentage_mutation):
    number_of_mutated_solutions = int(percentage_mutation * len(population))
    mutated_solutions = []
    nb_neurons = [10, 12, 14, 16]
    nb_layers = [1, 2, 3, 4, 5, 6, 7, 8]
    batch_size = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    optimizer = ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']
    activation = ['relu', 'elu', 'tanh', 'sigmoid']
    mapping_dict = {0: 'nb_neurons', 1: 'nb_layers', 2: 'batch_size', 3: 'optimizer', 4: 'activation'}
    for i in range(number_of_mutated_solutions):
        parent = random.choice(population)
        index = randint(0, len(parent) - 1)
        parent[index] = random.choice(eval(mapping_dict[index]))
        mutated_solutions.append(parent)
    return mutated_solutions


# Method to select the elit population

def select_elit_population(population, crossover_paths, mutated_paths, population_size, epochs, x_train, y_train,
                           x_test, y_test, fitness_dictionary, solution_dictionary, accuracy_dictionary):
    grouped_population = population + crossover_paths + mutated_paths
    fitness_cross, solution_cross, accuracy_cross = create_dictionary_with_percentages_of_cumulated_fitness(
        crossover_paths, epochs, x_train, y_train, x_test, y_test)
    fitness_mutation, solution_mutation, accuracy_mutation = create_dictionary_with_percentages_of_cumulated_fitness(
        crossover_paths, epochs, x_train, y_train, x_test, y_test)

    fitness = dict(collections.ChainMap(fitness_dictionary, fitness_cross, fitness_mutation))
    solution = dict(collections.ChainMap(solution_dictionary, solution_cross, solution_mutation))
    accuracy = dict(collections.ChainMap(accuracy_dictionary, accuracy_cross, accuracy_mutation))

    elit_keys = sorted(accuracy, key=accuracy.get, reverse=False)[:population_size]
    fitness_dict = dict((k, fitness[k]) for k in elit_keys)
    solution_dict = dict((k, solution[k]) for k in elit_keys)
    accuracy_dict = dict((k, accuracy[k]) for k in elit_keys)
    elit_population = [solution_dict[key] for key in elit_keys]

    return elit_population, fitness_dict, solution_dict, accuracy_dict