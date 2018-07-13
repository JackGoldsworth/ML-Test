import math
import quandl
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression


# Sentdex ML tutorial p.2 - 4
class StockRegression:
    # Getting the data from Quandl
    data = quandl.get("WIKI/GOOGL")
    # Getting data needed for percents
    data = data[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
    # H/L percent
    data["HL_PCT"] = (data["Adj. High"] - data["Adj. Close"]) / data["Adj. Close"] * 100
    # Percent change
    data["PCT_change"] = (data["Adj. Close"] - data["Adj. Open"]) / data["Adj. Open"] * 100
    # Variables we need after getting percents
    data = data[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

    forecast_col = "Adj. Close"
    data.fillna(-99999, inplace=True)

    # The price label looking forward
    forecast_out = int(math.ceil(0.01*len(data)))

    data["label"] = data[forecast_col].shift(-forecast_out)
    data.dropna(inplace=True)

    x = np.array(data.drop(["label"], 1))
    y = np.array(data["label"])
    x = preprocessing.scale(x)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=.02)
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    print(accuracy)
