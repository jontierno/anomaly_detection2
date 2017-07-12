import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import seaborn as sns # for intractve graphs

from imblearn.under_sampling import CondensedNearestNeighbour
def plot_counts(normal, fraud):

    N = 5
    men_means = (normal)
    men_std = (2)

    ind = np.arange(1)  # the x locations for the groups
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, men_means, width, fill=False, hatch='\\', yerr=men_std, label='Normal')

    women_means = (fraud)
    women_std = (2)
    rects2 = ax.bar(ind + width, women_means, width, fill=False, hatch='/', yerr=women_std, label='Fraud')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Log(Counts)')
    ax.set_title('Records count by class')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels((''))
    ax.set_facecolor("white")

    ax.legend()


def get_set():
    credit_cards=pd.read_csv('../dataset/creditcard.csv')
    columns=credit_cards.columns
    features_columns=columns.delete(len(columns)-1)
    features_columns= features_columns.delete(0)
    x=credit_cards[features_columns]
    y=credit_cards['Class']
    Fraud_transacation = credit_cards[credit_cards["Class"] == 1]
    Normal_transacation = credit_cards[credit_cards["Class"] == 0]
    plot_counts(np.math.log(len(Normal_transacation)), np.math.log(len(Fraud_transacation)))




    return train_test_split(x, y, test_size=0.2, random_state=10)

def oversample(x,y):
    # Usamos SMOTE (Synthetic Minority Over Sampling Technique) para generar mas casos de fraude
    sampler=SMOTE(random_state=10)
    new_x,new_y=sampler.fit_sample(x,y)
    return new_x, new_y

def undersample(x,y):
    #sampler = InstanceHardnessThreshold(random_state=42)
    sampler = SMOTEENN(random_state=42)
    new_x,new_y=sampler.fit_sample(x,y)
    return new_x, new_y

def create_forest(estimators, x,y):
    forest = RandomForestClassifier(n_estimators=estimators, random_state=10)
    forest.fit(x,y)
    return forest

def test_forest(forest,x,y):
    result = forest.predict(x)
    return result, confusion_matrix(y,result)

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    classes = [0,1]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, cmap="coolwarm_r", annot=True, linewidths=0.5)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion(confusion):
    plt.figure()
    plot_confusion_matrix(confusion, title='Confusion matrix')
    plt.show()


def evaluate_model(x_train,x_test,y_train, y_test,trees = 10):
    forest = create_forest(trees, x_train,y_train)
    y_predict, confusion = test_forest(forest,x_test,y_test)
    print ("================== RESULTADOS ==================")
    print ("CONFUSION MATRIX ")
    print (confusion)
    tp = confusion[0][0]
    fp = confusion[1][0]
    fn = confusion[0][1]
    tn = confusion[1][1]
    print "TRUE NORMAL {}: ".format(tp)
    print "FALSE NORMAL {}: ". format(fp)
    print "FALSE FRAUD {}: ".format(fn)
    print "TRUE FRAUD {}: ".format(tn)
    print "PRECISION : {}".format(precision_score(y_test, y_predict))
    print "RECALL : {}".format(recall_score(y_test, y_predict))
    print "F1 : {}".format(f1_score(y_test, y_predict))
    plot_confusion(confusion)


def main():
    #normal model
    x_train, x_test, y_train, y_test = get_set()
    evaluate_model(x_train.as_matrix(),x_test,y_train.as_matrix(),y_test)
    #oversampled
    x_train_ov, y_train_ov = oversample(x_train, y_train)
    evaluate_model(x_train_ov,x_test,y_train_ov,y_test)
    #over under
    x_train_us, y_train_us = undersample(x_train, y_train)
    evaluate_model(x_train_us,x_test, y_train_us,y_test)


if __name__ == '__main__':
    main()
