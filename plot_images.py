import math

import matplotlib.pyplot as plt

def plot_tolmax(iterations):
    num_iter = [i + 1 for i in range(len(iterations))]
    iterations = [elem for elem in iterations]
    plt.xlabel("Номер итерации")
    plt.ylabel("Значение распознающего функционала")
    plt.plot(num_iter, iterations)
    plt.show()

