import numpy as np
from matplotlib import pyplot as plt


def sinus_noised(x_min, x_max, samples):
    population = [np.random.uniform(x_min, x_max) for _ in range(samples)]
    population.sort()
    sinus_values = [np.sin(x) for x in population]
    noised_values = [np.random.normal(x, 0.1) for x in sinus_values]
    return population, sinus_values, noised_values


def michalewicz(xx):
    x = xx[0]
    y = xx[1]
    sum = 0
    sum += np.sin(x) * (np.sin(1 * x**2 / np.pi)) ** (2*10)
    sum += np.sin(y) * (np.sin(2 * y**2 / np.pi)) ** (2*10)
    return -sum


def eggholder(xx):
    x = xx[0]
    y = xx[1]
    return -(y + 47) * np.sin(np.sqrt(np.abs(y + x/2 + 47))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))


def michalewicz_noised(x_min, x_max, y_min, y_max, samples):
    population_x = [np.random.uniform(x_min, x_max) for _ in range(samples)]
    population_y = [np.random.uniform(y_min, y_max) for _ in range(samples)]
    michalewicz_values = [michalewicz([population_x[i], population_y[i]]) for i in range(samples)]
    noised_values = [np.random.normal(x, 0.1) for x in michalewicz_values]
    return population_x, population_y, michalewicz_values, noised_values


def eggholder_noised(x_min, x_max, y_min, y_max, samples):
    population_x = [np.random.uniform(x_min, x_max) for _ in range(samples)]
    population_y = [np.random.uniform(y_min, y_max) for _ in range(samples)]
    eggholder_values = [eggholder([population_x[i], population_y[i]]) for i in range(samples)]
    noised_values = [np.random.normal(x, 10) for x in eggholder_values]
    return population_x, population_y, eggholder_values, noised_values
