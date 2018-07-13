import datetime
import math
import quandl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import pickle


# Sentdex ML tutorial p.2 - 6
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
    forecast_out = int(math.ceil(0.01 * len(data)))

    data["label"] = data[forecast_col].shift(-forecast_out)

    x = np.array(data.drop(['label'], 1))
    x = preprocessing.scale(x)
    x_lately = x[-forecast_out:]
    x = x[:-forecast_out:]
    data.dropna(inplace=True)

    y = np.array(data["label"])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=.02)
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    with open("data.pickle", "wb") as f:
        pickle.dump(clf, f)

    pickle_in = open("data.pickle", "rb")
    clf = pickle.load(pickle_in)
    accuracy = clf.score(x_test, y_test)

    forecast_set = clf.predict(x_lately)

    data['Forecast'] = np.nan

    last_date = data.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        data.loc[next_date] = [np.nan for _ in range(len(data.columns) - 1)] + [i]

    data['Adj. Close'].plot()
    data['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
