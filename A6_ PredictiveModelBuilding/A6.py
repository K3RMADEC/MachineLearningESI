#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:37:03 2017

@author: Raúl Gallego de la Sacristana y Sergio Alises Mendiola
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import preprocessing 
from scipy.stats.stats import pearsonr
import codecs
import csv

from tabulate import tabulate
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
from sklearn import neighbors
# 0.1 load data

def load_data(file, filter_parameters = None, excludes_features = None,
              outliers = None,city_params=None,bandera=None):
    
    
    df = pd.read_csv(file)
    df.replace('', np.nan, inplace=True, regex=True)
    if excludes_features != None:
        df.drop(labels = excludes_features,axis=1, inplace = True)
    
    city = df['city'] == city_params
    df = df[city]

    if city_params=='sj' and bandera is False:
        df.drop(df.index[87], inplace=True)
        df.drop(df.index[138], inplace=True)
        df.drop(df.index[397], inplace=True)
        df.drop(df.index[448], inplace=True)
        df.drop(df.index[707], inplace=True)
        df.drop(df.index[758], inplace=True)
        df.drop(df.index[701], inplace=True)
    if city_params=='iq' and bandera is False:
        df.drop(df.index[182], inplace=True)
        df.drop(df.index[233], inplace=True)
        df.drop(df.index[440], inplace=True)
        df.drop(df.index[491], inplace=True)
    return df
def plotdata(data,labels,name):
    fig, ax = plt.subplots()
    
    plt.scatter([row[0] for row in data], [row[1] for row in data], c=labels)
    ax.grid(True)
    fig.tight_layout()
    plt.title(name)
    plt.show()
    
def main():
    # 0. Load data
    excludes_features = ['week_start_date']
    file="dengue_features_train.csv"
    file_labels="dengue_labels_train.csv"
    
    data_train_sj=load_data(file,excludes_features=excludes_features,city_params="sj",bandera=False)
    data_train_iq=load_data(file,excludes_features=excludes_features,city_params="iq",bandera=False)
    
    data_labels_sj=load_data(file_labels,city_params='sj',bandera=False)
    data_labels_iq=load_data(file_labels,city_params='iq',bandera=False)
    
    
    cross(data_train_sj,data_labels_sj,"sj",18)
    cross(data_train_iq,data_labels_iq,"iq",20)
    
  
def cross(data_train,data_labels,ciudad,n):
    mergedf = pd.merge(data_train, data_labels, on = ['city', 'year', 'weekofyear'], how = 'outer')

    #Remove no useful features in standardization
    data = mergedf.drop(['city', 'year'], axis = 1, inplace = False)
   
    
    if ciudad=="sj":
		data=data[['ndvi_se', 'station_min_temp_c','weekofyear','total_cases']]
		X = data[ ['ndvi_se', 'station_min_temp_c','weekofyear']]
    else:
		data=data[['reanalysis_avg_temp_k', 'station_min_temp_c','station_precip_mm','total_cases']]
		X = data[ ['reanalysis_avg_temp_k', 'station_min_temp_c','station_precip_mm']]

    y = data['total_cases']
    

    for i, weights in enumerate(['uniform', 'distance']):
        total_scores = []
        for n_neighbors in range(1,30):
            knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            knn.fit(X,y)
            scores = -cross_val_score(knn, X,y,scoring='neg_mean_absolute_error', cv=10)
            total_scores.append(scores.mean())
        
       	plt.plot(range(0,len(total_scores)), total_scores, marker='o', label=weights)
        plt.ylabel('cv score')

	plt.legend()
    plt.show() 
    model(n,X,y,ciudad)

def model(n_neighbors,X,y,ciudad):
    #Entrenando el modelo 
    model=None
    for i, weights in enumerate(['uniform', 'distance']):
		knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
		model=knn.fit(X,y)		
		y_pred = model.predict(X)
		plt.plot(y, c='k', label='data')
		plt.plot(y_pred, c='g', label='prediction')
		plt.axis('tight')
		plt.legend()
		plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,weights))
		plt.show()
    
    #Prediccion
    excludes_features = ['week_start_date']
    datos_test=load_data("dengue_features_test.csv",excludes_features=excludes_features,city_params=ciudad,bandera=True)
    c=datos_test.drop(['city', 'year'], axis = 1, inplace = False)

    if ciudad=="sj":
		c=c[['ndvi_se', 'station_min_temp_c','weekofyear']]
    else:
		c=c[['reanalysis_avg_temp_k', 'station_min_temp_c','station_precip_mm']]

    knn_dos = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
    datos_pre = knn_dos.fit(X,y).predict(c)
    
    # write the results in a csv file
    f =codecs.open(ciudad+"Resultados.csv",'wt')
    data2=[]
    
    for j in range(0,len(datos_test)):
        row=[]
        row.append(datos_test.iloc[j]['city'])
        row.append(datos_test.iloc[j]['year'])
        row.append(datos_test.iloc[j]['weekofyear'])
        row.append(int(datos_pre[j]))
        data2.append(row)

    try:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(('city','year','weekofyear','total_cases'))
        writer.writerows(data2)
    finally:
        f.close()
   

if __name__ == '__main__':
    main()
    
    
