'''
Name: Dwipam Katariya, Neelam Tikone

This program executes best models that were tunned using K-fold validation for different parameters
for following Machine Learning algorithms:
    1. Gradient Boost Machine
    2. Support Vector Machine
    3. Decision Tree(ID3)
We have kept just tunning loop for Decision tree, rest all are with the best parameters.
    Algorithm   AUROC
    GBM         0.80
    SVM         0.78
    ID3         0.62
To execute this program run for target variable Drunk Driver execute following line:
    python cluster_analysis.py Kmode3.csv ',' drunkDr GBM
    where,
    Kmodes3.csv - Labeled data with the clusters.
    drunkDr - Variable name as dependent variable.
    GBM - Gradient Boost Machine 
'''



import graphlab as gl
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.cross_validation import StratifiedShuffleSplit
from imblearn.combine import SMOTETomek
from sklearn import svm
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
h2o.init()


class modelObj(object):
    def __init__(self,model,auc,hyperparam):
        self.model = model
        self.auc = auc
        self.hyperparam = hyperparam
    def get_auc(self):
        return self.auc
def dec_classifier():
    train_data,test_data = read()
    train_data = gl.SFrame(balance(train_data))
    models = {}
    for depth in range(1,21):
    
        folds = gl.cross_validation.KFold(train_data,5)

        for train, valid in folds:
            model = gl.decision_tree_classifier.create(train, target = sys.argv[3],max_depth = depth)
            pred = model.predict_topk(valid,output_type='probability',k=1)
            ac = metrics.auc(valid[sys.argv[3]].to_numpy(),pred['probability'].to_numpy(),reorder=True)
            print model.evaluate(valid)
            if depth in models.keys() and models[depth].auc<ac:
                models[depth].auc = ac
            elif depth not in models.keys():
                models[depth] = modelObj(model,ac,depth)
    print map(lambda x:x.get_auc(),models.values())
    best_model = max(models.values(),key = lambda x:x.auc)
    print best_model.model.evaluate(test_data)

def stratified_split(sf, target, train_size = 0.8, seed = None):
    sf = sf.add_row_number('row_no')
    index = StratifiedShuffleSplit(sf[target], n_iter = 1, test_size = 1-train_size, random_state = seed)
    split = []
    for train_index, test_index in index:
        split.append(sf[sf.apply(lambda x: x['row_no'] in list(train_index))])
        split.append(sf[sf.apply(lambda x: x['row_no'] in list(test_index))])
    return split

def balance(data):
    Y = data[sys.argv[3]]
    data = data.to_dataframe()
    X = data.ix[:,data.columns!=sys.argv[3]]
    cols = X.columns
    sm = SMOTETomek()
    X,Y =  sm.fit_sample(X,Y)
    X = pd.DataFrame(X,columns = cols)
    Y = pd.DataFrame(Y,columns = [sys.argv[3]])
    return X.join(Y,how='inner')
def svm_classifier():
    train_data,test_data = read()
    Y_train = (train_data[sys.argv[3]]).to_numpy()
    cols = train_data.column_names()
    train_data = (train_data.remove_columns([sys.argv[3]])).to_numpy()
    Y_test = (test_data[sys.argv[3]]).to_numpy() 
    test_data = (test_data.remove_columns([sys.argv[3]])).to_numpy()
    model = svm.SVC()
    model.fit(train_data,Y_train)

    pass 
def gradient_boosted_trees():
    train_data,test_data = read()
    train_data = balance(train_data)
    train_data = h2o.H2OFrame(train_data)
    test_data = h2o.H2OFrame(test_data.to_dataframe())
    train_data[sys.argv[3]] = train_data[sys.argv[3]].asfactor()
    test_data[sys.argv[3]] = test_data[sys.argv[3]].asfactor()
    model = H2OGradientBoostingEstimator(distribution = 'bernoulli', ntrees = 1000, max_depth = 10, learn_rate = 0.1,nfolds = 8)
    print(model.train(y = sys.argv[3],training_frame=train_data))
    print(model.model_performance(test_data))
    pass
def read():
    data = pd.read_csv(sys.argv[1],sep = sys.argv[2])
    data = gl.SFrame(data)
    data.remove_column('A_POSBAC')
    train_data,test_data = stratified_split(data,'drunkDr')
    train_data.remove_column('row_no')
    test_data.remove_column('row_no')
    train_data.materialize()
    test_data.materialize()
    return(train_data,test_data)


if sys.argv[4]=='GBM':
    gradient_boosted_trees()
if sys.argv[4] =='DTREE':
    dec_classifier()
if sys.argv[4] == 'SVM':
    svm_classifier()
