#!/usr/bin/python

import pickle
import sys

import matplotlib.pyplot
import numpy as np
from sklearn import linear_model

sys.path.append("../tools/")
from feature_format import featureFormat

# read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
data_dict.pop('TOTAL')
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
data = data[(data[:, 0] < 1000000)]
data = data[(data[:, 1] < 7000000)]

# data = data[data[:,1].argsort()]
# data = data[:np.math.floor(len(data) * 0.99).__int__(), ]

salary, bonus = zip(*data)
salary = np.reshape(salary, (-1, 1))
bonus = np.reshape(bonus, (-1, 1))

# your code below
reg = linear_model.LinearRegression()

reg.fit(salary, bonus)
print(reg.intercept_)
print(reg.coef_)
print(reg.score(salary, bonus))

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
