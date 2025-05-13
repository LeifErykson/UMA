import numpy as np
import random
import time
from matplotlib import pyplot as plt
from roulette_tree import RegressionTree
from generate_data import eggholder, eggholder_noised

MAX_DEPTH = 200
MIN_SAMPLES_SPLIT = 3
NUM_SAMPLES = 500


def main():
    random.seed(15)
    np.random.seed(15)

    pop_x, pop_y, _, noised = eggholder_noised(-600, 600, -600, 600, NUM_SAMPLES)

    X = np.column_stack((pop_x, pop_y))
    y = np.array(noised)

    start = time.time()
    regressor = RegressionTree(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, selection_method='roulette')
    regressor.train(X, y)
    train_time1 = round(time.time() - start, 3)

    figure = plt.figure()
    axis = figure.add_subplot(projection="3d")
    x1 = np.linspace(-600, 600, 100)
    x2 = np.linspace(-600, 600, 100)
    x1, x2 = np.meshgrid(x1, x2)
    results = eggholder([x1, x2])

    # axis.plot_surface(x1, x2, results, cmap='viridis')
    x1 = np.linspace(-600, 600, 100)
    x2 = np.linspace(-600, 600, 100)
    x1, x2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack((x1.flatten(), x2.flatten()))

    predictions = regressor.predict(X_grid)

    predictions = predictions.reshape(x1.shape)

    axis.plot_surface(x1, x2, predictions, cmap='plasma')

    plt.title(f"Roulette\nMax depth = {MAX_DEPTH}\n Min samples split = {MIN_SAMPLES_SPLIT}\n Samples = {NUM_SAMPLES}\nTrain time roulette: {train_time1}s")
    # plt.title("Eggholder funciton")
    plt.show()


if __name__ == "__main__":
    main()
