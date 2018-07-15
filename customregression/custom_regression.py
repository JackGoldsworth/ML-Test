import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from customregression.regression_util import RegressionUtil


class CustomRegression:

    def main(self):
        style.use("fivethirtyeight")
        x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        y = np.array([5, 4, 2, 1, 6, 7], dtype=np.float64)

        util = RegressionUtil()
        m = util.slope(x, y)
        b = util.intercept(x, y)

        regression_line = [(m * x) + b for x in x]

        plt.scatter(x, y)
        plt.plot(x, regression_line)
        plt.show()
