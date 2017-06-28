import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

url = "dataset/creditcard.csv"
names = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
         "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Class"]
dataset = pd.read_csv(url, names=names)

rng = np.random.RandomState(42)
dataset=dataset[["V14","V10","V12","V17","V4","Class"]]
array = dataset.values
X = array[:, 0:5]
Y = array[:, 5]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=rng)


print "\t*********************************** Random Forest ****************************"

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, Y_train)
Y_rf_pred = rf.predict(X_test)
cnf_matrix = confusion_matrix(Y_test, Y_rf_pred)
print cnf_matrix

tp = cnf_matrix[1][1]
fp = cnf_matrix[0][1]
tn = cnf_matrix[0][0]
fn = cnf_matrix[1][0]
precision = float(tp + tn) / float(tp + tn)
print '\tAccuracy: ' + str(
    np.round(precision_score(Y_test, Y_rf_pred), 4)) + '%'
recall = float(tp) / float(tp + fn)
print '\tRecall: ' + str(
    np.round(recall_score(Y_test, Y_rf_pred), 4)) + '%'
print '\tF1 score: ' + str(
    np.round(f1_score(Y_test, Y_rf_pred), 4)) + '%'


