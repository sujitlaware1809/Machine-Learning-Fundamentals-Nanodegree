# Report: Predict Bike Sharing Demand with AutoGluon Solution
This project focused on predicting bike sharing demand using AutoGluon within the AWS SageMaker Studio environment. We explored the influence of various factors on model performance through experimentation with three different models, all evaluated on the Kaggle platform.

#### NAME HERE
Sujit Laware

## Initial Training

### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
When the predictions were submitted to kaggle, they were rejected each time they contain negative value. So we needed to check negative value, and replace them with zero.

### What was the top ranked model that performed?
The top ranked model was the first one: it was trained using the default parameters of AutoGluon and got a score of 1.8 from Kaggle.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
EDA showed us the distribution of the numerical features, which allowed us to spot the skew in windspeed, registered, casual and count. Another action we did take is to engineer new features (year, month, day, hour) based on the feature datetime, in order to try improving our model results.

### How much better did your model preform after adding additional features and why do you think that is?
After having added our new features (year, month, day, hour) and removed the feature datetime (since its information is already conveyed in the engineered features), the new model kaggle score dropped to 0.47450, againt 1.81618 when the datetime column was used instead of the engineered ones (let us take note of the fact that datetime column is of type datetime).
So we can infer from this that AutoGluon has the hability to preprocess and engineer datetime features in different ways and select the best one, in order to allow the model to get the best result.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
After tuning the model hyper-parameters, the models performance dropped to 1.31230 against 1.81618 when we left the hyper-parameters untouched.

### If you were given more time with this dataset, where do you think you would spend more time?
If given more time, I will spend more time on the first model, the baseline one (its hyper-parameters were not tuned, and datetime features were not engineered in the data fed to it). That choice is based on the fact that clearly, AutoGluon is very good at predicting well on our dataset without our intervention, so we should give it more time to train (for e.g., doubling the training duration and observing the results).

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|600|best_quality|NaN|1.81618|
|add_features|600|best_quality|NaN|0.47450|
|hpo|12x60|best_quality|nn:activation -- dropout_prob; gmb:num_boost_round -- num_leaves; scheduler; searcher|1.31230|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](https://i.postimg.cc/8czRCtj6/model-train-score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](https://i.postimg.cc/dQNdBRh5/model-test-score.png)

## Summary

This report details an exploration of bike-sharing demand prediction using AutoGluon within AWS SageMaker Studio. The project involved training and evaluating models on the Kaggle platform.


