from __future__ import division
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, StratifiedShuffleSplit, cross_val_predict,train_test_split
from sklearn.metrics.classification import classification_report
from sklearn import decomposition, preprocessing


# In[]
# Loading Data
data = pd.read_csv("dados.csv", header=0, delimiter=";")
data = data.fillna(-1)

for l in ['end_esc', 'end_al', ]:
    aux = preprocessing.LabelEncoder().fit_transform([str(i) for i in data[l]])    
    data[l] = aux


# Indexing the data

X = data.drop('sit_al', axis=1)
y = data['sit_al']

# In[]
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_r = pca.transform(X)
y_r = y.values.ravel()

target_names=['0','1','-']
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y_r == i, 0], X_r[y_r == i, 1], color=color, alpha=.3, lw=i+1,
                label=target_name)
pl.legend(loc='best', shadow=False, scatterpoints=1)
pl.title('PCA of IRIS dataset')
pl.show()
# In[]

#for k in ['nivel', 'turno', 'etnia', 'resp']:
#    sns.factorplot    

# In[]
n_runs=1
for run in range(0,n_runs):
   random_seed=run
   np.random.seed(random_seed)
   
   clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', )
   #clf = GradientBoostingClassifier(n_estimators=10, )
   
   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=int(random_seed))
   y_p  = cross_val_predict(clf,X.squeeze(), y.squeeze(), cv=cv, n_jobs=-1)
   
   print(classification_report(y_p, y))
   
   clf.fit(X,y)
   n=len(clf.feature_importances_); 
   pl.bar(range(n), clf.feature_importances_); 
   pl.xticks(np.array(range(n))+0.5,X.columns.values, rotation=90)
   pl.show()
   
# In[]   