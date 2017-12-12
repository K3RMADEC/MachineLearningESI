# Feature Engineering

## Load Data

In this activity we use all the data set for the first time, we separate the data of different cities.

## Correlation 

We extract the correlation between each feature and label (total_cases).

![Correlation: Feature - Total cases](correlation.png)

The next thing is to select the features with the most correlation (R>0.75):

|  |  |
| -- | -- |
| __Feature__ | __R value__ |
| ndvi_nw | 0.75077 |
| reanalysis_dew_point_temp_k |  0.834108 |
| reanalysis_precip_amt_kg_per_m2 | 0.753697 |
| station_avg_temp_c |  0.796544 |
| station_min_temp_c | 0.888011 |
</center>

## Cross Validation

We need to select the best maximun depth, that's why we will execute a loop that creates a decision tree with each value between 2 and 30.

![Cross Validation](cross_validation.png)

The best maximun depth is 2.

## Build the model

We create the regression model with the MSE (Mean Squared Error) criterion and the best maximun depth.

### Decision Tree

![Decision Tree](tree.png)

### Features Relevances

|  |  |
| -- | -- |
| __Feature__ | __Relevance__ |
| ndvi_nw | 0.854556 |
| reanalysis_dew_point_temp_k |  0.145444 |
| reanalysis_precip_amt_kg_per_m2 | 0 |
| station_avg_temp_c |  0 |
| station_min_temp_c | 0 |
</center>

Features with more relevance: ['ndvi_nw', 'reanalysis_dew_point_temp_k']
