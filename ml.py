import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import time
from datetime import datetime
import itertools

import pickle

def class_labels(age):
    if 1 <= age <= 2:
        return 1
    elif 3 <= age <= 9:
        return 2
    elif 10 <= age <= 20:
        return 3
    elif 21 <= age <= 25:
        return 4
    elif 26 <= age <= 27:
        return 5
    elif 28 <= age <= 31:
        return 6
    elif 32 <= age <= 36:
        return 7
    elif 37 <= age <= 45:
        return 8
    elif 46 <= age <= 54:
        return 9
    elif 55 <= age <= 65:
        return 10
    else:
        return 11

train = np.load("/teamspace/studios/this_studio/canny_train.npy")
test = np.load("/teamspace/studios/this_studio/canny_test.npy")

# /teamspace/studios/this_studio/canny_test.npy
feature_names = pd.read_csv("/teamspace/studios/this_studio/canny_features_names.csv")

train_df = pd.DataFrame(train, columns=feature_names["canny_edge_features"])
test_df = pd.DataFrame(test, columns=feature_names["canny_edge_features"])


train_df['age'] = train_df['age'].astype(np.uint8)
test_df['age'] = test_df['age'].astype(np.uint8)

train_df['label'] = train_df['age'].map(class_labels)
test_df['label'] = test_df['age'].map(class_labels)




## Training Phase

X_train = train_df.drop(columns=['age', 'label'])
y_train = train_df['label']

X_test = test_df.drop(columns=['age', 'label'])
y_test = test_df['label']
     

# Scaling X_train to the standard scale.
ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
     

# Transforming X_test to the same scale.
X_test_sc = ss.transform(X_test)


# rfc = RandomForestClassifier(
#                              ccp_alpha=0,
#                              min_samples_split=2,
#                              min_samples_leaf=1,
#                              random_state=42
#                             )
     
# rfc_params = {'n_estimators' : [50, 100, 200],
#               'max_depth' : [5, 7, 9],
#              }


# rfc_gs = GridSearchCV(rfc, param_grid=rfc_params, n_jobs=-1, cv=5)

# rfc_gs.fit(X_train_sc, y_train)

# print(rfc_gs.best_params_)
# print(rfc_gs.best_score_)

# Creating a SVC object.
svc = SVC(random_state=42)
     

# Establishing ranges of hyperparameters of SVC for GridSearchCV.

svc_params = {'C' : [0.001, 1],
              'kernel' : ['rbf', 'poly', 'linear'],
              'degree' : [3, 5]
             }
     

# Creating a GridSearchCV object for the SVC object defined above.

svc_gs = GridSearchCV(svc, param_grid=svc_params, n_jobs=-1, cv=5)
     

# Fitting X_train_sc and y_train on GridSearchCV object with SVC defined above.

svc_gs.fit(X_train_sc, y_train)

# Training Accuracy

svc_train_acc = svc_gs.score(X_train_sc, y_train)
svc_test_acc = svc_gs.score(X_test_sc, y_test)
     

# Actual Testing Accuracy

svc_test_acc = svc_gs.score(X_test_sc, y_test)


print("SVC summary of accuracy scores:")
print(f"GridSearchCV best accuracy = {round(svc_gs.best_score_, 3)}")
print("\nUsing GridSearchCV best params suggested,")
print(f"Training accuracy = {round(svc_train_acc, 3)}")
print(f"Testing accuracy = {round(svc_test_acc, 3)}")
