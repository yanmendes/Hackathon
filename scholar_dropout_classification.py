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

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# In[]
# Loading Data
data = pd.read_csv("bruto.csv", header=0, delimiter=",")
#data = data.fillna(-1)
data = data.query('sit_al == "Ativo" or sit_al == "Inativo"')

target_var=['sit_al']
drop_var=['end_al', 'bairro_al']
count_var=['etnia','turno']

# In[]
for t in target_var:
    for k in count_var:
        print('-'*32)
        for d,df in data.groupby(k): 
            a=df[t].value_counts()
            s= a[1]/a[0]*100 if len(a)>1 else 0
            print(d,'\t\t',s)
            
        print('-'*32)
# In[]
for t in target_var:
    for k in count_var:
        g=sns.countplot(x=k, hue=t, data=data)
        pl.xticks(rotation=90)
        pl.show()
        
        
# In[]
for l in data.columns:
    aux = preprocessing.LabelEncoder().fit_transform([str(i) for i in data[l]])    
    data[l] = aux

# Indexing the data
X = data.drop(target_var + drop_var, axis=1)
y = data['sit_al']

# In[]
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_r = pca.transform(X)
y_r = y.values.ravel()

target_names=['0','1','-']
colors = ['darkorange', 'navy', 'turquoise', ]
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y_r == i, 0], X_r[y_r == i, 1], color=color, alpha=.3, lw=i+1,
                label=target_name)
pl.legend(loc='best', shadow=False, scatterpoints=1)
pl.title('PCA')
pl.show() 

# In[]
n_runs=1
for run in range(0,n_runs):
   random_seed=run
   np.random.seed(random_seed)
   
   clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', )
   
   SSS = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=run)
   for train_index, test_index in SSS.split(X.values, y.values):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        print(classification_report(y_pred, y_test))
   
#   n=len(clf.feature_importances_); 
#   pl.bar(range(n), clf.feature_importances_); 
#   pl.xticks(np.array(range(n))+0.5,X.columns.values, rotation=90)
#   pl.show()
  
  
# In[]   
n_runs=1
for run in range(0,n_runs):
   random_seed=run
   np.random.seed(random_seed)
   
   clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', )

   SSS = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=run)
   for train_index, test_index in SSS.split(X.values, y.values):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Apply the random under-sampling
        rus = RandomUnderSampler(return_indices=True)
        X_resampled, y_resampled, idx_resampled = rus.fit_sample(X_train, y_train)

        clf.fit(X_resampled,y_resampled)
        y_pred=clf.predict(X_test)
        print(classification_report(y_pred, y_test))


#   n=len(clf.feature_importances_); 
#   pl.bar(range(n), clf.feature_importances_); 
#   pl.xticks(np.array(range(n))+0.5,X.columns.values, rotation=90)
#   pl.show()   

# In[]   
n_runs=1
for run in range(0,n_runs):
   random_seed=run
   np.random.seed(random_seed)
   
   clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', )

   SSS = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=run)
   for train_index, test_index in SSS.split(X.values, y.values):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Apply the random over-sampling
        ros = RandomOverSampler()
        X_resampled, y_resampled = ros.fit_sample(X_train, y_train)

        clf.fit(X_resampled,y_resampled)
        y_pred=clf.predict(X_test)
        print(classification_report(y_pred, y_test))
                

#   n=len(clf.feature_importances_); 
#   pl.bar(range(n), clf.feature_importances_); 
#   pl.xticks(np.array(range(n))+0.5,X.columns.values, rotation=90)
#   pl.show()   

# In[]

