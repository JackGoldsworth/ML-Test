from statistics import mean


class RegressionUtil:


    def slope(self, x, y):
        return ((mean(x) * mean(y)) - mean(x * y)) / ((mean(x) * mean(x)) - mean(x ** 2))

    def intercept(self, x, y):
        return mean(y) - (self.slope(x, y) * mean(x))