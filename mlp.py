from __future__ import division
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, fbeta_score
from sklearn import tree
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
import csv
import time
import itertools

# Loading Data
data = pd.read_csv("dados.csv", header=0, delimiter=";")
data = data.fillna(0)

# Indexing the data
data = data[data.sit_al != 4]
del data['end_esc']
del data['end_al']

y = data['sit_al']
del data['sit_al']

y = list(map(lambda x: not x, y))

with open('output.csv', 'w', 1) as csvfile:
	for attrs in range(1, 11):
		spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['Combination', 'Precision', 'Recall', 'F-Measure'])
		combinations = list(itertools.combinations(data, attrs))
		for combination in combinations:
			newdf = data[list(combination)]
			X_train, X_test, y_train, y_test = train_test_split(newdf, y, test_size=0.33)
			clf = RandomForestClassifier(n_estimators=20, class_weight='balanced')
			clf.fit(X_train, y_train)
			predicted = clf.predict(X_test)
			precision = precision_score(y_test, predicted)
			recall = recall_score(y_test, predicted)
			fmeasure = fbeta_score(y_test, predicted, 2)
			spamwriter.writerow([str(combination), precision, recall, fmeasure])
			#i_tree = 0
			#for tree_in_forest in clf.estimators_:
				#with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
				#	my_file = tree.export_graphviz(tree_in_forest, out_file = my_file, feature_names=newdf.columns, filled=True, rounded=True)
				#os.system('dot -Tpng tree_' + str(i_tree) + '.dot -o tree.png')
				#i_tree = i_tree + 1
