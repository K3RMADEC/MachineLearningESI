#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:12:32 2017

@author: Sergio Alises Mendiola and Raúl Gallego de la Sacristana Alises. 
"""
from sklearn import preprocessing 
from sklearn.decomposition import PCA
import codecs
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn import metrics

def plotdata(data,labels,name):
    fig, ax = plt.subplots()
    
    plt.scatter([row[0] for row in data], [row[1] for row in data], c=labels)
    ax.grid(True)
    fig.tight_layout()
    plt.title(name)
    plt.show()


def load_data(path):
        
    f = codecs.open(path, "r", "utf-8")
    print(f)
    names = []
    count = 0
    for line in f:

        if count > 0:
    		# remove double quotes
            row = line.split(";")
            row.pop(0)
            row.pop(0)
            row.pop(0)
            row.pop(0)
            if row != []:
                data = [float(el) for el in row]
                names.append(data)
        count += 1
    return names


data=load_data("../Data/dengue_features_train.csv")
#1. Data normalizazion
#http://scikit-learn.org/stable/modules/preprocessing.html
min_max_scaler = preprocessing.MinMaxScaler()
datanorm = min_max_scaler.fit_transform(data)

#2. Principal Component Analysis
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(datanorm)

print(estimator.explained_variance_ratio_)

#3. Check the best value of k
silhouettes = []
for i in range(2,10):
    centroidsplus, labelsplus, zplus =  sklearn.cluster.k_means(X_pca, i, init="k-means++" )
    silhouettes.append(metrics.silhouette_score(datanorm, labelsplus))
    
plt.plot(range(2,10), silhouettes, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette')
plt.show()

# Clustering with k-means++
k=2
centroidsplus, labelsplus, zplus =  sklearn.cluster.k_means(X_pca, k, init="k-means++" )
plotdata(X_pca,labelsplus,'K-Means++')