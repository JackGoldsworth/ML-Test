from statistics import mean
import random
import numpy as np


class RegressionUtil:

    def slope(self, x, y):
        return ((mean(x) * mean(y)) - mean(x * y)) / ((mean(x) * mean(x)) - mean(x ** 2))

    def intercept(self, x, y):
        return mean(y) - (self.slope(x, y) * mean(x))

    def squared_error(self, y_point, y_line):
        return sum((y_point - y_line) ** 2)

    # Coefficient of Determination
    def cod(self, y_point, y_line):
        y_mean_line = [mean(y_point) for y in y_point]
        squared_error = self.squared_error(y_point, y_line)
        squared_error_mean = self.squared_error(y_point, y_mean_line)
        return 1 - squared_error / squared_error_mean

    # Just creates a fake data set.
    def create_dataset(self, hm, variance, step=2, correlation=False):
        val = 1
        ys = []
        for i in range(hm):
            y = val + random.randrange(-variance, variance)
            ys.append(y)
            if correlation and correlation == 'pos':
                val += step
            elif correlation and correlation == 'neg':
                val -= step

        xs = [i for i in range(len(ys))]
        return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
