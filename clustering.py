import pandas as pd
import numpy as np
from kmodes import kmodes
from sklearn.preprocessing import OneHotEncoder as ohe
import sys
from collections import Counter


def kmodesClust(data,targetVariable,cluster):
    kmodeObj = kmodes.KModes(n_clusters = cluster, init = 'Huang', n_init = 50,  max_iter =400)
    model = kmodeObj.fit(np.array(data))
    clusters = model.predict(np.array(data))
    data = mergeclusters(data,clusters,'clusters')
    return (data,model)

def mergeclusters(data,clusters,name):
    clusters = pd.DataFrame(clusters, index = data.index.values,columns=[name])
    data = data.join(clusters,how='inner') 
    return data

def analysis_kmodes(data,targetVariable):
    if targetVariable != None:
        tgt =  data[targetVariable]
        del data[targetVariable]
    dist = []; cost = []
    for cluster in range(3,4):
        print "Clusters: ", cluster
        X,model = kmodesClust(data,targetVariable,cluster)
        mergeclusters(X,tgt,'drunkDr').to_csv('Kmode3.csv')
        print model.cost_
        dist.append(distance_measure(X,model.cluster_centroids_))
        cost.append(model.cost_)
    return (dist,cost)

#Distance Measure for K-modes is average distance for clusters as per "Huang" distance measure. 

def distance_measure(data,centroid):
    clusterData = []
    for clust in set(np.array(data['clusters'])):
        cdist = []
        cdata = data[data['clusters'] == clust]
        dist = map(lambda x:dict(x),map(lambda x:Counter(cdata[x]),cdata.columns))
        for i in dist:
            for j in i:
                i[j] = i[j]*1.0/len(cdata)
            tot_dist = []
        for row in range(len(cdata)):
            row = np.array(cdata.iloc[row])
            di_ = []
            for x in range(len(row)-1):
                if centroid[row[len(row)-1]][x]==row[x]:
                    di_.append(1)
                else:
                    di_.append(1-dist[x][row[x]])
            tot_dist.append(sum(di_))
        cdist.append(np.mean(tot_dist))
    print np.mean(cdist)        
    return np.mean(cdist)

def kmeansClust(data,targetVariable):
    #1. Convert categorical data to one-hot encoding
    #2. remove durnken_dr variable from clustering
    #3. Output data with labeled clusters
    pass

def analysis_kmeans(data,targetVariable):
    data = kmeansClust(data,targetVariable)
    #1. Compute Sum of Squared distance
    #2. plot how increase in K changes in Mean sum of sqaures
    #3. Plot how runing multiple interations for best K returns changes in cluster
    #Return best clustered data.
    pass

def read_data():
    data = pd.read_csv('filtered_fatalities.csv',sep=",")
    data = data.reset_index()
    data = data.drop(['index'],axis=1)
    return data

def main():
    data = read_data()
    kmeans_result = analysis_kmeans(data, sys.argv[1])
    kmodes_result = analysis_kmodes(data, sys.argv[1])
if __name__=='__main__':main()
