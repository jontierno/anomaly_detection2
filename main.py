# Load dataset


# Load libraries
import pandas
import collections
from sklearn import model_selection
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

url = "dataset/creditcard.csv"
names = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
         "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Class"]
dataset = pandas.read_csv(url, names=names)


nobel_dataset = dataset[dataset['Class'] == 0]
outliers_dataset = dataset[dataset['Class'] == 1]

rng = np.random.RandomState(42)
# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(5,6), sharex=False, sharey=False)
# plt.show()
array = nobel_dataset.values
X_nobel = array[:, 1:30]
Y_nobel = array[:, 30]
validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_nobel, Y_nobel, test_size=validation_size,random_state=rng)

array_out = outliers_dataset.values
X_outlier = array_out [:, 1:30]
Y_outlier = array_out [:, 30]




clf = IsolationForest()
clf.fit(X_train, Y_train)


#Y_train = Y_train * -2  + 1
#Y_validation = Y_validation * -2  + 1



y_pred_valid = clf.predict(X_validation)
y_pred_outlier = clf.predict(X_outlier)



c1 = collections.Counter(y_pred_valid)
c2 = collections.Counter(y_pred_outlier)

tp = c1[1]
fn = c1[-1]

tn = c2[-1]
fp = c2[1]


Y_total_pred = np.concatenate((y_pred_valid, y_pred_outlier), axis=0)
Y_total_truth = np.concatenate((Y_validation, Y_outlier), axis=0)
print "\t True Positive {}, False negative {}, True Negative {}, False positive {}".format(tp, fn, tn, fp)


avg ='macro'




#print "\tY_train composition: %s" % precision_score(Y_total_truth, Y_total_pred, average=avg)
#print "\tY_validation composition: %s" % recall_score(Y_total_truth, Y_total_pred, average=avg)
#print "\ty_pred_train composition: %s" % f1_score(Y_total_truth, Y_total_pred, average=avg)
#print "\ty_pred_validation composition: %s" % collections.Counter(y_pred_validation)

#target_names = ['Fraudes', 'Validas']
#print classification_report(Y_validation, y_pred_validation, target_names=target_names)

print confusion_matrix(Y_total_truth, Y_total_pred)
print "\tPrecision: %1.7f" % precision_score(Y_validation, Y_total_pred)
print "\tRecall: %1.7f" % recall_score(Y_validation, Y_total_pred)
print "\tF1: %1.7f\n" % f1_score(Y_validation, Y_total_pred)
#print "\tMatrix: %s\n" % str(cnf_matrix)







