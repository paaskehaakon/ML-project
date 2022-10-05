import numpy as np 
import pandas as pd 

class LogisticRegression:
    def __init__(self):
        #theta of the model(weights)
        self.w = None

        self.bias = 0

    def update_parameters(self, new_weight, new_bias):
        self.w = self.w - self.learning_rate * new_weight
        self.bias = self.bias - self.learning_rate * new_bias

    def from_lin_to_logistic(self, X):
        # create linear model (Y = Ax + b)
        lin_mod = np.dot(X, self.w) + self.bias
        # run linear model through sigmoid function
        predict_y = sigmoid(lin_mod)
        return predict_y

    def fit(self, X, y):
        self.learning_rate = 0.5 # experimented with learning rate. found 0.5 to give good results even though it is very high
        self.number_of_iterations = 1000 # could probably do with a lot fewer iterations. 
        
        #initialize parameters
        number_of_samples, number_of_features = X.shape
        self.w = np.zeros(number_of_features)

        #perform iterations
        for iteration in range(self.number_of_iterations):
            predict_y = self.from_lin_to_logistic(X)
            # apply the gradiant descent formula
            derivative_w = np.dot(X.T, (predict_y - y))*(1 / number_of_samples) 
            derivative_bias = np.sum(predict_y - y)*(1 / number_of_samples) 

            # update weight and bias
            self.update_parameters(derivative_w, derivative_bias)


    def predict(self, X):
        #create linear model (Y = Ax + b)
        lin_mod = np.dot(X, self.w) + self.bias
        # pass the linear model thorugh the sigmoid function
        predict_y = sigmoid(lin_mod)
        # assign prediction 1 if over 0.5. 0 otherwise
        predict_y_binary = [1 if i > 0.5 else 0 for i in predict_y]
        return np.array(predict_y_binary)


    

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
