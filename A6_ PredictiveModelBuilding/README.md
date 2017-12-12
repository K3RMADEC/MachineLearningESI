# Predictive Model Bulding

In this task we will obtain the first results. For this purpose we will use KNN algorithm to obtain predictions.

We will use the more relevant features of each city for build the model with the same dataset that the previus task.

## Cross Validation

As we mentioned before, we will use KNN algorithm. Before of this we will obtain the best number of neighbours for each city.


* __San Juan:__

![San Juan CV][1] 


* __Iquitos:__

![Iquitos CV][2]

## Build the model

When we have the best number of neighbours we build the model using 'uniform' and 'distance' parameters.

* __San Juan:__

![San Juan Uniform][3] 

![San Juan Distance][4] 

* __Iquitos:__

![Iquitos Uniform][6]

![Iquitos Distance][5]

We can see that 'distance' parameter is better than 'uniform', so in this task we will use 'distance'.

## Prediction

We load the test file 'dengue features' in the models building in previously for predict the total cases. Then we save this results in csv format and upload in the competition.

We obtain: 27.2212 (Score) and 749/2384 (Rank)


[1]:https://github.com/K3RMADEC/MachineLearningESI/blob/master/A6_%20PredictiveModelBuilding/Images/corrQui.png
[2]:https://github.com/K3RMADEC/MachineLearningESI/blob/master/A6_%20PredictiveModelBuilding/Images/corr_sj.png
[3]:https://github.com/K3RMADEC/MachineLearningESI/blob/master/A6_%20PredictiveModelBuilding/Images/trainSj.png
[4]:https://github.com/K3RMADEC/MachineLearningESI/blob/master/A6_%20PredictiveModelBuilding/Images/trainSj_distance.png
[5]:https://github.com/K3RMADEC/MachineLearningESI/blob/master/A6_%20PredictiveModelBuilding/Images/train_qui_distance.png
[6]:https://github.com/K3RMADEC/MachineLearningESI/blob/master/A6_%20PredictiveModelBuilding/Images/train_qui_uniform.png
