from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *


# Method to plot the evolution over time in the GUI

def plot_evolution_over_time_in_gui(error_over_time, main_frame, root):
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
