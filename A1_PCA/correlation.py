# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:55:12 2017

@authors: Sergio.Alises and Raul.Gallego
"""

import codecs
from numpy import corrcoef, transpose, arange
from pylab import pcolor, show, colorbar, xticks, yticks
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# 0. Load Data
f = codecs.open("../Data/dengue_features_train_csv.csv", "r", "utf-8")
semana = []
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
			semana.append(map(float, row))
    count += 1



# plotting the correlation matrix
#http://glowingpython.blogspot.com.es/2012/10/visualizing-correlation-matrices.html
R = corrcoef(transpose(semana))
pcolor(R)
colorbar()
yticks(arange(0,20),range(0,20))
xticks(arange(0,20),range(0,20))
show()


# http://stanford.edu/~mwaskom/software/seaborn/examples/many_pairwise_correlations.html
# Generate a mask for the upper triangle
sns.set(style="white")
mask = np.zeros_like(R, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(200, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(R, mask=mask, cmap=cmap, vmax=.8,
            square=True, xticklabels=2, yticklabels=2,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)