from __future__ import division
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, fbeta_score
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
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

def nivel(x):
	if(x == 'EDUCACAO INFANTIL'):
		return 0
	elif(x == 'ENSINO FUNDAMENTAL' or x == 'PRESENCIAL - ENSINO FUNDAMENTAL' or x == 'Ensino Fundamental por Ciclos/Ano'):
		return 1
	elif(x == 'PRESENCIAL - ENSINO MEDIO'):
		return 2
	else:
		return x

def turno(x):
	if(x == 'MANHA'):
		return 0
	elif(x == 'INTERMEDIARIO 1' or x == 'MANHA - TARDE'):
		return 1
	elif(x == 'TARDE'):
		return 2
	elif(x == 'VESPERTINO' or x == 'TARDE - NOITE' or x == 'INTERMEDIARIO 2'):
		return 3
	elif(x == 'NOITE'):
		return 4
	elif(x == 'INTEGRAL'):
		return 5
	elif(x == 'INTERMEDIARIO 4'):
		return 4
	else:
		return x

def tipo_atendimento(x):
	if(x == 'REGULAR'):
		return 0
	elif(x == 'TURMAS DE ATENDIMENTO AEE'):
		return 1
	elif(x == 'TURMAS DE ATIVIDADE COMPLEMENTAR'):
		return 2
	else:
		return x

def etnia(x):
	if(x == 'Branca'):
		return 0
	elif(x == 'Nao declarada'):
		return 1
	elif(x == 'Preta'):
		return 2
	elif(x == 'Parda'):
		return 3
	elif(x == 'Indigena'):
		return 4
	elif(x == 'Amarela'):
		return 5
	else:
		return x

def repetente(x):
	if(x == 'NOVATO'):
		return 0
	elif(x == 'REPETENTE'):
		return 1
	else:
		return x

def sit_al(x):
	if(x == 'Ativo' or x =='Remanejado'):
		return 0
	elif(x == 'Inativo' or x == 'Remanejado e Inativo'):
		return 1
	else:
		return x

def resp(x):
	if(x == 'Mae' or x =='Pai' or x == 'MAE' or x == 'PAI'):
		return 1
	elif(x == 'Madrasta' or x == 'Padrasto' or x == 'Irmao' or x == 'Irma'):
		return 2
	elif(x == 'Avo' or x == 'Tio(a)'):
		return 3
	elif(x == 'Outros' or x == 'Vizinho(a)'):
		return 4
	elif(x == 'Proprio Aluno'):
		return 0
	else:
		return x

def necessidades(x):
	if(x):
		return 1
	else:
		return 0

def transp(x):
	if(x == 'Nao Utiliza' or x == 'Nao Informado'):
		return 1
	else:
		return 0

# Loading Data
data = pd.read_csv("bruto.csv", header=0, delimiter=",")
data = data.fillna(0)

#Pre-processamento nivel
data = data[data.nivel != 'CRECHE']
data['nivel'] = data.nivel.map(nivel)

#Pre-processamento turno
data['turno'] = data.turno.map(turno)

#Pre-processamento tipo_atendimento
data['tipo_atendimento'] = data.tipo_atendimento.map(tipo_atendimento)

#Pre-processamento etnia
data['etnia'] = data.etnia.map(etnia)

#Pre-processamento repetente
data['repetente'] = data.repetente.map(repetente)

#Pre-processamento sit_al
data['sit_al'] = data.sit_al.map(sit_al)

#Pre-processamento resp
data['resp'] = data.resp.map(resp)

#Pre-processamento necessidades
data['necessidades'] = data.necessidades.map(necessidades)

#Pre-processamento transp
data['transp'] = data.transp.map(transp)

#Pre-processamento end aluno
le = preprocessing.LabelEncoder()
le.fit(data['end_al'])
data['end_al'] = le.transform(data['end_al'])

X = data.values
y = np.asarray(data['sit_al'])
del data['necessidades']
del data['sit_al']

with open('output.csv', 'w', 1) as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(['Combination', 'Precision Avg', 'Precision Std', 'Recall Avg', 'Recall Std', 'F-Measure Avg', 'F-Measure Std'])
	skf = StratifiedKFold(n_splits=5)
	precision = list()
	recall = list()
	fmeasure = list()
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
		clf.fit(X_train, y_train)
		predicted = clf.predict(X_test)
		precision.append(precision_score(y_test, predicted))
		recall.append(recall_score(y_test, predicted))
		fmeasure.append(fbeta_score(y_test, predicted, 2))
	spamwriter.writerow([0, np.mean(precision), np.std(precision), np.mean(recall), np.std(recall), np.mean(fmeasure), np.std(fmeasure)])
