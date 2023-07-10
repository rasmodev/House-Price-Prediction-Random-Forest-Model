# House-Price-Prediction-Random-Forest-Model
This repository contains a machine learning algorithm that trains a Random Forest model to predict house prices based on specified features of the homes, using the California Housing Dataset. The dataset used to train and evaluate the Random Forest model to predict median housing prices. The Random Forest model is trained using cross-validation and hyperparameter tuning to optimize its performance. The best model is then evaluated on the test set to assess its performance.

# Data Preparation
## 1.1 Read Data
This section imports the necessary libraries (pandas, numpy, and matplotlib) and loads the data from a CSV file called 'housing.csv'.
It then splits the data into training and testing sets using the train_test_split function from scikit-learn.

The California Housing Prices dataset has a total of 20,640 records and 9 features. The dataset is split into training and testing sets with an 80:20 ratio, and a random state of 42.

## 1.2 Data Cleaning
This section checks for missing values in the dataset, and since the 'total_bedrooms' column has missing values, it uses the SimpleImputer method to impute the missing values in that column with the median value in the column. It then creates a copy of the training and testing data without the text attribute 'ocean_proximity' and imputes the missing values using the trained imputer.

## 1.3 Feature Engineering
This section creates new features or transforms existing features to better represent the underlying patterns in the data. It creates three new features: 'rooms_per_household', 'bedrooms_per_room', and 'population_per_household', and applies them to the training and testing data.

## 1.4 Feature Scaling
This section scales the features in the dataset to a standard scale using the StandardScaler method from scikit-learn. It creates two pipelines: one for the training data and another for the testing data. The pipelines include the imputer and scaler for the numerical features, and the OneHotEncoder for the categorical feature 'ocean_proximity'.

# Modelling
This section trains a Random Forest model using the training data and predicts the median house values for the testing data. It then calculates the mean squared error for the predicted values using the actual values. Finally, it uses cross-validation to estimate the performance of the model.

## Model Training
The model is trained using a Random Forest regression algorithm. The code for training the model is in the 2. Model Training section of the notebook. The hyperparameters of the model are set using cross-validation, and the performance of the model is evaluated using the mean squared error (MSE) metric. The function display_scores() is used to display the cross-validation scores for the model.

## Model Tuning
The code for tuning the model is in the 3. Model Tuning section of the notebook. The hyperparameters of the model are tuned using a randomized search with 50 iterations. The best hyperparameters are chosen based on the MSE metric. The RandomizedSearchCV class from the Scikit-learn library is used to perform the randomized search.

## Model Testing
The code for testing the model is in the 4. Model Testing section of the notebook. The performance of the model is evaluated on the testing set using the root mean squared error (RMSE) metric. The model is also evaluated using cross-validation. Finally, the model is saved to a file using the pickle library.

# Compare Models
The RMSE and cross-validation scores of the Random Forest model and other regression models such as Decision Tree Regressor, K-Nearest Neighbors Regressor, Artificial Neural Networks (ANN), and Ensemble Methods such as AdaBoost, XGBoost, and LightGBM are compared in the 5. Compare Models section of the notebook. The Random Forest model outperforms the other models in terms of the RMSE and the cross-validation scores.

# Next Steps
Further exploration and improvement of the Random Forest model could include tuning additional hyperparameters, exploring feature importance, and investigating ensemble techniques such as stacking or blending to enhance the predictive performance.
