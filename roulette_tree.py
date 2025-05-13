import numpy as np
from generate_data import sinus_noised, michalewicz_noised, michalewicz, eggholder, eggholder_noised
import random
import matplotlib.pyplot as plt

# random_choice function to choose randomly between many options with many different probabilities
def random_choice(options, probabilities):
    if len(options) != len(probabilities):
        raise ValueError("The number of options and probabilities should be equal.")

    return random.choices(options, probabilities)[0]

class Node:
    def __init__(self, feature_idx=None, threshold=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value
        self.left = None
        self.right = None

class RegressionTree:
    def __init__(self, max_depth=None, min_samples_split=2, selection_method='classic'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.selection_method = selection_method
        self.root = None

    # Mean squared error
    def mse(self, values):
        return np.mean((values - np.mean(values))**2)

    def split(self, args, values, feature_idx, threshold):
        left_mask = args[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return args[left_mask], args[right_mask], values[left_mask], values[right_mask]

    def calculate_best_idx(self, thresholds, values):
        n = len(thresholds)
        idx = 0
        sos_thresholds = []
        threshold_sum = 0

        while idx < n:
            current_noise = values[idx]
            for value in values:
                # Summing up squared diff
                threshold_sum += (value - current_noise) ** 2
            sos_thresholds.append(threshold_sum)
            threshold_sum = 0
            idx += 1
        # Roulette
        if self.selection_method == 'roulette':
            max_threshold = max(thresholds)
            wages = [(max_threshold / x) for x in thresholds]
            sum_wages = sum(wages)
            probabilities = [wage / sum_wages for wage in wages]
            return random_choice([x for x in range(len(thresholds))], probabilities)
        else:
            # Classic
            return np.argmin(sos_thresholds)

    def find_best_split(self, args, values):
        best_mse = float('inf')
        best_feature_idx = None
        best_threshold = None
        mses = []

        for feature_idx in range(args.shape[1]):
            thresholds = np.unique(args[:, feature_idx])
            values_copy = values.copy()
            best_idx = self.calculate_best_idx(thresholds, values_copy)
            best_threshold = thresholds[best_idx]
            _, _, values_left, values_right = self.split(args, values, feature_idx, best_threshold)
            if len(values_left) != 0:
                mse_left = self.mse(values_left)
            else:
                mse_left = 0
            if len(values_right) != 0:
                mse_right = self.mse(values_right)
            else:
                mse_right = 0
            mse = (mse_left * len(values_left) + mse_right * len(values_right)) / len(values)
            mses.append(mse)
            # It will be useful if classic mode
            if mse < best_mse:
                best_mse = mse
                best_feature_idx = feature_idx
                # print("I got there")

        # Roulette
        if self.selection_method == 'roulette':
            max_mse = max(mses)
            wages = [(max_mse / x) for x in mses]
            sum_wages = sum(wages)
            probabilities = [wage / sum_wages for wage in wages]
            best_feature_idx  = random_choice([x for x in range(len(mses))], probabilities)
            thresholds = np.unique(args[:, feature_idx])
            values_copy = values.copy()
            best_idx = self.calculate_best_idx(thresholds, values)
            best_threshold = thresholds[best_idx]

        return best_feature_idx, best_threshold

    def build_tree(self, args, values, depth):
        # stop building new nodes conditions #1
        if depth == self.max_depth or len(args) < self.min_samples_split or np.all(values == values[0]):
            value = np.mean(values)
            return Node(value=value)
        # 2
        feature_idx, threshold = self.find_best_split(args, values)
        if feature_idx is None:
            value = np.mean(values)
            return Node(value=value)

        args_left, args_right, values_left, values_right = self.split(args, values, feature_idx, threshold)

        node = Node(feature_idx=feature_idx, threshold=threshold)
        node.left = self.build_tree(args_left, values_left, depth + 1)
        node.right = self.build_tree(args_right, values_right, depth + 1)

        return node

    def train(self, args, values):
        self.root = self.build_tree(args, values, 0)

    # getting value from the tree
    def predict_recursive(self, node, arg):
        if node.value is not None:
            return node.value

        if arg[node.feature_idx] <= node.threshold:
            return self.predict_recursive(node.left, arg)
        else:
            return self.predict_recursive(node.right, arg)

    def predict(self, args):
        predictions = []
        for arg in args:
            prediction = self.predict_recursive(self.root, arg)
            predictions.append(prediction)
        return np.array(predictions)
