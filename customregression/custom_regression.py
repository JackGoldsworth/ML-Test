from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use("fivethirtyeight")
x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
y = np.array([5, 4, 2, 1, 6, 7], dtype=np.float64)


def slope(x, y):
    return ((mean(x) * mean(y)) - mean(x * y)) / ((mean(x) * mean(x)) - mean(x ** 2))


def intercept(x, y):
    return mean(y) - (slope(x, y) * mean(x))


m = slope(x, y)
b = intercept(x, y)

regression_line = [(m * x) + b for x in x]

plt.scatter(x, y)
plt.plot(x, regression_line)
plt.show()
