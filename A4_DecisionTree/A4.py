#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:37:03 2017

@author: Raúl Gallego de la Sacristana y Sergio Alises Mendiola
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing 
from scipy.stats.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz 
from tabulate import tabulate
from sklearn.cross_validation import cross_val_score

# 0.1 load data

def load_data(file, filter_parameters = None, excludes_features = None,
              outliers = None):
    
    
    
    df = pd.read_csv(file)
    df.replace('', np.nan, inplace=True, regex=True)
    if excludes_features != None:
        df.drop(labels = excludes_features,axis=1, inplace = True)
    
    return df

def normalization_with_minmax(data):
    
    min_max_scaler = preprocessing.MinMaxScaler()
        
    return min_max_scaler.fit_transform(data.dropna().values)  

def main():
    # 0. Load data
    excludes_features = ['week_start_date']
    file="Data/dengue_features_train.csv"
    data_train=load_data(file,excludes_features=excludes_features)

    data_labels = load_data(
            file="Data/dengue_labels_train.csv")
    

    #Datos unidos.
    mergedf = pd.merge(data_train, data_labels, on = ['city', 'year', 'weekofyear'], how = 'outer')
    
      
    #Remove no useful features in standardization
    mergedf_drop = mergedf.drop(['city', 'year', 'weekofyear'], axis = 1, inplace = False)
    features= mergedf_drop.columns.tolist()[:-1]
    norm_df = normalization_with_minmax(mergedf_drop)
    
       # 1. Correlation
    corr = map(lambda x1: pearsonr(x1,norm_df[-1].tolist())[0],
               [norm_df[x].tolist() for x in range(len(features))])
    
    y_pos = np.arange(len(features))
    
    plt.bar(y_pos, corr, align='center')
    plt.xticks(y_pos, features, rotation = 90)
    plt.ylabel('Correlation')
    plt.title('Correlation features vs totalcase')
    plt.show()

    features_corr = zip(features, corr)
    print ''
    print tabulate(features_corr, headers = ['Feature','R value'])
    features_selected = [features_corr[i][0] for i in range(len(features_corr))
                            if abs(features_corr[i][1]) > 0.75]
    
       # 2. CROSS VALIDATION ANALYSIS
    df_dropna = mergedf_drop.dropna()
    total_scores = []

    for i in range(2, 30):
        regressor = DecisionTreeRegressor(criterion = 'mse', max_depth=i)
        regressor.fit(df_dropna[features_selected], df_dropna['total_cases'])
        scores = -cross_val_score(regressor, df_dropna[features_selected],
                 df_dropna['total_cases'], scoring='neg_mean_absolute_error', cv=10)
        total_scores.append((scores.mean(), scores.std()))
    
    scores_mean = [total_scores[i][0] for i in range(len(total_scores))]
    
    plt.plot(range(2,30), scores_mean, marker='o')
    plt.xlabel('max_depth')
    plt.ylabel('cv score')
    plt.show()
    
    info_regression = [(i+2, total_scores[i][0], '+/- '+str(total_scores[i][1])) for i in range(len(total_scores))]
    print ''
    print tabulate(info_regression, headers = ['Level depth', 'Mean','Standard Deviation'])
    best_max_depth = total_scores.index(min(total_scores))+2
    print '\nBest MAX_DEPTH: %d' % (best_max_depth)
    
    #3. Build the model
    
    #3.1 Model Parametrization 
    # criterion: mse mean squared error, which is equal to variance reduction as feature selection criterion
    # splitter: best/random
    # max_depth: low value avoid overfitting
    
    regressor = DecisionTreeRegressor(criterion = 'mse', max_depth = best_max_depth)

    #3.2 Model construction
    regressor.fit(df_dropna[features_selected], df_dropna['total_cases'])
    
    
    #3.3.  Model Visualization
    
    dot_data = export_graphviz(regressor, out_file="out.dot", 
                               feature_names = features_selected,
                               filled=True, rounded=True,  
                               special_characters=True)  
    
    graph = graphviz.Source(dot_data, format = 'png')   
    #graph.render('decision_tree_mse', 'images', cleanup= True)
    
    # 3.4 Feature Relevances
    
    print '\n\t\t[ RELEVANCES FEATURES (MSE) ]\n'
    relevances_list = zip(features_selected, regressor.feature_importances_)
    
    print tabulate(relevances_list, headers=['Feature selected', 'Relevance'])
    
    reg_values = list(regressor.feature_importances_)
    reg_values_sort = sorted(reg_values, reverse = True)
    
    max_values_feature = []
    
    for i in range(2):
        pos = reg_values.index(reg_values_sort[i])
        max_values_feature.append(features_selected[pos])
    
    print '\nFeatures with more relevance: \n\t' + str(max_values_feature)
    
if __name__ == '__main__':
    main()
    
    
