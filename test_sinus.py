import numpy as np
import random
import time
from matplotlib import pyplot as plt
from roulette_tree import RegressionTree
from generate_data import sinus_noised

MAX_DEPTH = 200
MIN_SAMPLES_SPLIT = 3
NUM_SAMPLES = 500


def main():
    random.seed(16)
    np.random.seed(16)

    pop, _, noised = sinus_noised(0, 10, NUM_SAMPLES)
    x = np.linspace(0, 10, 200)
    y = np.sin(x)
    plt.plot(x, y, "-", label="Sinus", alpha=0.3)
    plt.plot(pop, noised, "o", label="Noised values", alpha=0.3)

    start = time.time()
    regressor = RegressionTree(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, selection_method='classic')
    regressor.train(np.column_stack((pop,)), np.array(noised))
    train_time1 = round(time.time() - start, 3)

    predictions = regressor.predict(np.column_stack((x,)))
    plt.plot(x, predictions, "-", label="Predicted values classic")

    start = time.time()
    regressor = RegressionTree(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, selection_method='roulette')
    regressor.train(np.column_stack((pop,)), np.array(noised))
    train_time2 = round(time.time() - start, 3)

    predictions = regressor.predict(np.column_stack((x,)))
    plt.plot(x, predictions, "-", label="Predicted values roulette")
    plt.title(f"Max depth = {MAX_DEPTH}\n Min samples split = {MIN_SAMPLES_SPLIT}\n Samples = {NUM_SAMPLES}\n Train time classic: {train_time1}s\n Train time roulette: {train_time2}s")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
