import pandas as pd
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import numpy as np
from kmodes import kmodes
from sklearn.preprocessing import OneHotEncoder

import sys
import bpython

def kmodesClust(data,targetVariable):
    if targetVariable != None:
        tgt =  data[targetVariable]
        del data[targetVariable]
    for i in range(0,100):
        import bpdb;bpdb.set_trace()
        kmodeObj = kmodes.KModes(n_clusters = i, init = 'Huang', n_init = 100,  max_iter = 500)
        clusters = kmodeObj.fit_predict(np.array(data))
        data = mergeclusters(data,clusters)
    return data

def mergeclusters(data,clusters):
    clusters = pd.DataFrame(clusters, index = data.index.values)
    return data.join(clusters,how='inner')

def analysis_kmodes(data,targetVariable):
    data = kmodesClust(data,targetVariable)
    pass

def kmeansClust(data,targetVariable):
    if targetVariable != None:
        tgt =  data[targetVariable]
        del data[targetVariable]
    data = data.drop(['index'],axis=1)
    enc = OneHotEncoder()
    print("DIMENSIONS", data.shape)
    enc.fit(data)
    df= pd.DataFrame(enc.transform(data).toarray())
    print(df.head())
    '''for i in range(2,100):
        print("Clusters: ", i)
        kmeans = KMeans(n_clusters=i, random_state=0, max_iter = 300).fit(df)
        print("cluateres")
        labels = kmeans.labels_
        print("SILHOUTTE: ",metrics.silhouette_score(df, labels, metric='euclidean'))
        print("Calinski-Harabaz Index: ",metrics.calinski_harabaz_score(df, labels))'''
    kmeans = KMeans(n_clusters=3, max_iter = 300).fit(df)
    labels = kmeans.labels_
    data = mergeclusters(data,labels)
    data = mergeclusters(data,tgt)
    print("Centroids:  ",kmeans.cluster_centers_)
    centroids = pd.DataFrame(kmeans.cluster_centers_)
    centroids.to_csv("/Users/Neelam/Desktop/centroids.csv")
    data.to_csv("/Users/Neelam/Desktop/Kmeans3.csv")
    

def analysis_kmeans(data,targetVariable):
    data = kmeansClust(data,targetVariable)
    #1. Compute Sum of Squared distance
    #2. plot how increase in K changes in Mean sum of sqaures
    #3. Plot how runing multiple interations for best K returns changes in cluster
    #Return best clustered data.
    pass

def read_data():
    data = pd.read_csv('/Users/Neelam/Documents/FARS2015NationalCSV/filtered_fatalities.csv',sep=",")
    #print(data.columns.tolist())
    data = data.reset_index()
    return data

def main():
    data = read_data()
    kmeans_result = analysis_kmeans(data, sys.argv[1])
    #kmodes_result = analysis_kmodes(data, sys.argv[1])


if __name__=='__main__':main()
