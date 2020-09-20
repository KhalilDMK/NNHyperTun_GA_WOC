# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:00:34 2018

@author: K0DAMA01
"""

import time
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox
from Code.Data import housing_dataset
from Code.GA_utils import *
from Code.WOC_utils import *


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

# Main function

def main():
    epochs = 3
    b2 = False
    b3 = False
    b4 = False

    # Reading input file and creating the dataset

    filename = variable_filename.get()
    if filename == "Boston housing prices":
        dataset = housing_dataset()
        x_train, y_train, x_test, y_test = dataset.get_housing_dataset()

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
