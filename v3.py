#https://www.kaggle.com/gargmanish/how-to-handle-imbalance-data-study-in-detail
import warnings

import matplotlib.pyplot as plt # to plot graph
import numpy as np # for linear algebra
import pandas as pd # to import csv and for data manipulation
import seaborn as sns # for intractve graphs
from sklearn.cross_validation import train_test_split # to split the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler # for preprocessing the data

warnings.filterwarnings('ignore')
data = pd.read_csv("../creditcard.csv",header = 0)
data.info()

# Now lets check the class distributions
sns.countplot("Class",data=data)
Count_Normal_transacation = len(data[data["Class"]==0]) # normal transaction are repersented by 0
Count_Fraud_transacation = len(data[data["Class"]==1]) # fraud by 1
Percentage_of_Normal_transacation = float(Count_Normal_transacation)/(
    Count_Normal_transacation+Count_Fraud_transacation)
print("percentage of normal transacation is",Percentage_of_Normal_transacation*100)
Percentage_of_Fraud_transacation= float(Count_Fraud_transacation)/(Count_Normal_transacation+Count_Fraud_transacation)
print("percentage of fraud transacation",Percentage_of_Fraud_transacation*100)
Fraud_transacation = data[data["Class"]==1]
Normal_transacation= data[data["Class"]==0]
plt.figure(figsize=(10,6))
plt.subplot(121)
Fraud_transacation.Amount.plot.hist(title="Fraud Transacation")
plt.subplot(122)
Normal_transacation.Amount.plot.hist(title="Normal Transaction")
plt.show()

# the distribution for Normal transction is not clear and it seams that all transaction are less than 2.5 K
# So plot graph for same
Fraud_transacation = data[data["Class"]==1]
Normal_transacation= data[data["Class"]==0]
plt.figure(figsize=(10,6))
plt.subplot(121)
Fraud_transacation[Fraud_transacation["Amount"]<= 2500].Amount.plot.hist(title="Fraud Tranascation")
plt.subplot(122)
Normal_transacation[Normal_transacation["Amount"]<=2500].Amount.plot.hist(title="Normal Transaction")
plt.show()

# for undersampling we need a portion of majority class and will take whole data of minority class
# count fraud transaction is the total number of fraud transaction
# now lets us see the index of fraud cases
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = np.array(data[data.Class == 0].index)

# now let us a define a function for make undersample data with different proportion
# different proportion means with different proportion of normal classes of data
def undersample(normal_indices, fraud_indices, times):  # times denote the normal data = times*fraud data
    Normal_indices_undersample = np.array(
        np.random.choice(normal_indices, (times * Count_Fraud_transacation), replace=False))
    undersample_data = np.concatenate([fraud_indices, Normal_indices_undersample])
    undersample_data = data.iloc[undersample_data, :]

    print("the normal transacation proportion is :",
          float(len(undersample_data[undersample_data.Class == 0])) / len(undersample_data))
    print("the fraud transacation proportion is :",
          float(len(undersample_data[undersample_data.Class == 1])) / len(undersample_data))
    print("total number of record in resampled data is:", len(undersample_data))
    return (undersample_data)

## first make a model function for modeling with confusion matrix
def model(model,features_train,features_test,labels_train,labels_test):
    clf= model
    clf.fit(features_train,labels_train.values.ravel())
    pred=clf.predict(features_test)
    cnf_matrix=confusion_matrix(labels_test,pred)
    print("the recall for this model is :",float(cnf_matrix[1,1])/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    fig= plt.figure(figsize=(6,3))# to plot the graph
    print("TP",cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud
    print("TN",cnf_matrix[0,0]) # no. of normal transaction which are predited normal
    print("FP",cnf_matrix[0,1]) # no of normal transaction which are predicted fraud
    print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal
    sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test,pred))

def data_prepration(x): # preparing data for training and testing as we are going to use different data
    #again and again so make a function
    x_features= x.ix[:,x.columns != "Class"]
    x_labels=x.ix[:,x.columns=="Class"]
    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=0.3)
    print("length of training data")
    print(len(x_features_train))
    print("length of test data")
    print(len(x_features_test))
    return(x_features_train,x_features_test,x_labels_train,x_labels_test)

# before starting we should standridze our ampount column
data["Normalized Amount"] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data.drop(["Time","Amount"],axis=1,inplace=True)
data.head()


Undersample_data1_features_train,Undersample_data1_features_test,Undersample_data1_labels_train,\
Undersample_data1_labels_test = data_prepration(data)
clf= RandomForestClassifier(n_estimators=10)
model(clf,Undersample_data1_features_train,Undersample_data1_features_test,Undersample_data1_labels_train,Undersample_data1_labels_test)


#let us train this model using undersample data and test for the whole data test set
for i in range(1,4):
    print("the undersample data for {} proportion".format(i))
    print()
    Undersample_data = undersample(normal_indices,fraud_indices,i)
    print("------------------------------------------------------------")
    print()
    print("the model classification for {} proportion".format(i))
    print()
    undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test=data_prepration(Undersample_data)
    data_features_train,data_features_test,data_labels_train,data_labels_test=data_prepration(data)
    #the partion for whole data
    print()
    clf=RandomForestClassifier(n_estimators=100)
    model(clf,undersample_features_train,data_features_test,undersample_labels_train,data_labels_test)
    # here training for the undersample data but tatsing for whole data
    print("_________________________________________________________________________________________")

featimp = pd.Series(clf.feature_importances_,index=data_features_train.columns).sort_values(ascending=False)
print(featimp) # this is the property of Random Forest classifier that it provide us the importance
# of the features use

data1=data[["V14","V10","V12","V17","V4","Class"]]
data1.head()

Undersample_data1 = undersample(normal_indices,fraud_indices,1)
#only for 50 % proportion it means normal transaction and fraud transaction are equal so passing
Undersample_data1_features_train,Undersample_data1_features_test,Undersample_data1_labels_train,Undersample_data1_labels_test = data_prepration(Undersample_data1)
clf= RandomForestClassifier(n_estimators=100)
model(clf,Undersample_data1_features_train,Undersample_data1_features_test,Undersample_data1_labels_train,Undersample_data1_labels_test)

