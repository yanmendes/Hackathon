from __future__ import division
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn import tree
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
import csv
import time
import itertools
from sklearn.ensemble import RandomForestClassifier

# Loading Data
data = pd.read_csv("dados.csv", header=0, delimiter=";")
data = data.fillna(0)

# Indexing the data
data = data[data.sit_al != 4]
del data['end_esc']
del data['end_al']

y = data['sit_al']
del data['sit_al']

with open('output.csv', 'w', 1) as csvfile:
	for attrs in range(1, 2):
		spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['Combination', 'Score'])
		combinations = list(itertools.combinations(data, attrs))
		for combination in combinations:
			newdf = data[list(combination)]
			X_train, X_test, y_train, y_test = train_test_split(newdf, y, test_size=0.33)
			clf = RandomForestClassifier(n_estimators=20, class_weight='balanced_subsample')
			clf.fit(X_train, y_train)
			predicted = clf.predict(X_test)
			score = 0
			for i in range(0, len(predicted)):
				if(predicted[i] == list(y_test)[i]):
					score = score + 1
			spamwriter.writerow([str(combination), score / len(predicted)])
