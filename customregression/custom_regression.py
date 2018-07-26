import matplotlib.pyplot as plt
from matplotlib import style
from customregression.regression_util import RegressionUtil


# Uses fake data for now, and doesn't exactly build off of the error,
# but for now I'm just following tutorials so...
# Sentdex ML tutorial p.7 - 12
class CustomRegression:

    def main(self):
        style.use("fivethirtyeight")

        util = RegressionUtil()
        x, y = util.create_dataset(40, 40, 2, correlation='pos')
        m = util.slope(x, y)
        b = util.intercept(x, y)

        regression_line = [(m * x) + b for x in x]

        r_squared = util.cod(y, regression_line)

        predict_x = 5
        predict_y = (m * predict_x + b)
        print(r_squared)

        plt.scatter(x, y, label="data")
        plt.scatter(predict_x, predict_y, color="g")
        plt.plot(x, regression_line, label="regression line")
        plt.show()
