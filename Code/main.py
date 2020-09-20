# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:00:34 2018

@author: K0DAMA01
"""

import numpy as np
import time
from random import randint
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox
import collections
import math


# Method to get housing dataset and preprocess it

def get_housing_dataset():
    boston_housing = keras.datasets.boston_housing
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    order = np.argsort(np.random.random(y_train.shape))
    x_train = x_train[order]
    y_train = y_train[order]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, y_train, x_test, y_test


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


# Method to create and train the model_cost of solution

def cost_of_solution(solution, epochs, x_train, y_train, x_test, y_test):
    nb_neurons, nb_layers, batch_size, optimizer, activation = solution

    model = Sequential()
    model.add(Dense(nb_neurons, input_dim=13, activation=activation))
    for i in range(nb_layers):
        model.add(Dense(nb_neurons, activation=activation))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    model.fit(x_train, y_train, epochs=epochs, shuffle=True, batch_size=batch_size, verbose=0)
    score, error = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    keras.backend.clear_session()
    return error


# Method to create a dictionary of chromosomes with their finesses

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


# Method to plot the evolution over time in the GUI

def plot_evolution_over_time_in_gui(error_over_time):
    fig = Figure(figsize=(4.5, 3))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    for i in range(len(error_over_time)):
        ax.plot(error_over_time[i])
    graph = FigureCanvasTkAgg(fig, master=main_frame)
    canvas = graph.get_tk_widget()
    canvas.grid(row=1, column=3, rowspan=10, pady=5, sticky=E)
    fig.canvas.draw_idle()
    root.update()
    return fig, ax


# Method to get WOC solution

def get_woc_solution(best_solutions):
    woc_dictionary = []
    for i in range(len(best_solutions[0])):
        woc_dictionary.append(dict(
            (key, [best_solutions[j][i] for j in range(len(best_solutions))].count(key)) for key in
            set([best_solutions[j][i] for j in range(len(best_solutions))])))
    woc_solution = [max(dictionary, key=dictionary.get) for dictionary in woc_dictionary]
    return woc_solution


# Main function

def main():
    epochs = 3
    b2 = False
    b3 = False
    b4 = False

    # Reading input file and creating the dataset

    filename = variable_filename.get()
    if filename == "Boston housing prices":
        x_train, y_train, x_test, y_test = get_housing_dataset()

    # Getting population size

    population_size = entry_population_size.get()
    try:
        population_size = int(population_size)
        b2 = True
    except:
        messagebox.showerror("Error: Invalid population size",
                             "Invalid population size. Please enter an integer population size.")

    # Getting the number of iterations

    number_of_iterations = entry_number_iterations.get()
    try:
        number_of_iterations = int(number_of_iterations)
        b3 = True
    except:
        messagebox.showerror("Error: Invalid number of iterations",
                             "Invalid number of iterations. Please enter an integer number of iterations.")

    # Getting the number of solutions to combine

    number_of_solutions = entry_number_solutions.get()
    try:
        number_of_solutions = int(number_of_solutions)
        b4 = True
    except:
        messagebox.showerror("Error: Invalid number of solutions",
                             "Invalid number of solutions. Please enter an integer number of solutions.")

    # Getting the percentages of crossover and mutation

    percentage_crossover = scale_percentage_crossover.get() / 100
    print(percentage_crossover)
    percentage_mutation = scale_percentage_mutation.get() / 100
    print(percentage_mutation)

    # Getting the crossover method

    crossover_method = variable_crossover_method.get()

    if b2 == True and b3 == True and b4 == True:

        start_time = time.time()
        best_solutions = []
        cost_text = ''
        error_over_time = []

        for j in range(number_of_solutions):

            progress_bar_iteration.delete('1.0', END)
            progress_bar_iteration.insert(END, str(j + 1) + ' / ' + str(number_of_solutions))
            root.update()

            # Creating initial population

            min_costs_over_time = []
            avg = []

            population = create_initial_population(population_size)
            fitness_dictionary, solution_dictionary, accuracy_dictionnary = create_dictionary_with_percentages_of_cumulated_fitness(
                population, epochs, x_train, y_train, x_test, y_test)
            min_costs_over_time.append(accuracy_dictionnary[min(accuracy_dictionnary, key=accuracy_dictionnary.get)])
            print('solution ' + str(j + 1))

            for i in range(number_of_iterations):
                print('Generation ' + str(i + 1))

                # Crossover

                crossover_solutions = generate_crossover_solutions(population, percentage_crossover, crossover_method,
                                                                   epochs, x_train, y_train, x_test, y_test,
                                                                   fitness_dictionary, solution_dictionary)
                print('Crossover done: ' + str(len(crossover_solutions)) + ' created')

                # Mutation

                mutated_solutions = generate_mutated_solutions(population, percentage_mutation)
                print('Mutation done: ' + str(len(mutated_solutions)) + ' created')

                # Elitism

                population, fitness_dictionary, solution_dictionary, accuracy_dictionnary = select_elit_population(
                    population, crossover_solutions, mutated_solutions, population_size, epochs, x_train, y_train,
                    x_test, y_test, fitness_dictionary, solution_dictionary, accuracy_dictionnary)
                print('Elitism done: ' + str(len(population)) + ' selected')
                min_costs_over_time.append(
                    accuracy_dictionnary[min(accuracy_dictionnary, key=accuracy_dictionnary.get)])
                avg.append(float(sum(accuracy_dictionnary.values())) / len(accuracy_dictionnary))

            # Save best solution
            best_solutions.append(population[0])

            # Printing the evolution of the error over time

            print('min costs over time: ' + str(avg))
            error_over_time.append(avg)
            fig1, ax1 = plot_evolution_over_time_in_gui(error_over_time)

            # Filling the result text box

            cost_text += 'Solution ' + str(j + 1) + ' error: ' + str(round(avg[-1], 2)) + '\n'
            cost_time_text.delete('1.0', END)
            cost_time_text.insert(END, cost_text + 'Time: ' + str(round(time.time() - start_time, 2)) + ' s')
            root.update()

        # Wisdom of crowds
        woc_solution = get_woc_solution(best_solutions)
        woc_solution_cost = cost_of_solution(woc_solution, epochs, x_train, y_train, x_test, y_test)
        cost_text += 'WOC solution: ' + str(woc_solution) + ' Cost: ' + str(round(woc_solution_cost, 2)) + '\n'
        cost_time_text.delete('1.0', END)
        cost_time_text.insert(END, cost_text + 'Time: ' + str(round(time.time() - start_time, 2)) + ' s')
        root.update()


# Creating the GUI

root = Tk()
root.geometry('850x500')
root.title("Hyperparameter tuning with GA WOC")

main_frame = Frame(root)

label_filename = Label(main_frame, text="File name")
variable_filename = StringVar(main_frame)
variable_filename.set("Boston housing prices")
dropdown_filename = OptionMenu(main_frame, variable_filename, "Boston housing prices")

label_population_size = Label(main_frame, text="Population size")
entry_population_size = Entry(main_frame)

label_number_iterations = Label(main_frame, text="Number of iterations")
entry_number_iterations = Entry(main_frame)

label_number_solutions = Label(main_frame, text="Number of solutions to combine")
entry_number_solutions = Entry(main_frame)

label_percentage_crossover = Label(main_frame, text="Percentage crossover")
scale_percentage_crossover = Scale(main_frame, from_=1, to=100, orient=HORIZONTAL, length=250)
scale_percentage_crossover.set(70)

label_percentage_mutation = Label(main_frame, text="Percentage mutation")
scale_percentage_mutation = Scale(main_frame, from_=0.01, to=5, orient=HORIZONTAL, digits=5, resolution=0.01,
                                  length=250)
scale_percentage_mutation.set(4)

label_crossover_method = Label(main_frame, text="Crossover method")
variable_crossover_method = StringVar(main_frame)
variable_crossover_method.set("Random cutpoint")
dropdown_crossover_method = OptionMenu(main_frame, variable_crossover_method, "Random cutpoint",
                                       "Random section in middle")

button = Button(main_frame, text="Solve", command=main)

label_iteration = Label(main_frame, text="Progress")
progress_bar_iteration = Text(main_frame, height=1, width=15)

label_woc_solution = Label(main_frame, text="WOC solution")
cost_time_text = Text(main_frame, height=5, width=60)

main_frame.grid(row=0)
label_filename.grid(row=1, column=0, sticky=E, pady=5)
dropdown_filename.grid(row=1, column=1, pady=5, sticky=W)
label_population_size.grid(row=2, column=0, sticky=E, pady=5)
entry_population_size.grid(row=2, column=1, pady=5, sticky=W)
label_number_iterations.grid(row=3, column=0, sticky=E, pady=5)
entry_number_iterations.grid(row=3, column=1, pady=5, sticky=W)
label_number_solutions.grid(row=4, column=0, sticky=E, pady=5)
entry_number_solutions.grid(row=4, column=1, pady=5, sticky=W)
label_crossover_method.grid(row=5, sticky=E, pady=5)
dropdown_crossover_method.grid(row=5, column=1, pady=5, sticky=W)
label_percentage_crossover.grid(row=6, column=0, sticky=E, pady=5)
scale_percentage_crossover.grid(row=7, column=0, pady=5, sticky=W, columnspan=2)
label_percentage_mutation.grid(row=8, column=0, sticky=E, pady=5)
scale_percentage_mutation.grid(row=9, column=0, pady=5, sticky=W, columnspan=2)
button.grid(row=10, column=1, pady=5, sticky=W)
cost_time_text.grid(row=12, column=2, padx=5, pady=5, sticky=W, columnspan=2)
label_iteration.grid(row=1, column=2, sticky=E, pady=5)
progress_bar_iteration.grid(row=1, column=3, pady=5, sticky=W)
label_woc_solution.grid(row=10, column=2, pady=5, columnspan=2)

root.mainloop()
