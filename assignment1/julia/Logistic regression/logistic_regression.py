import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class LogisticRegression:
    
    def __init__(self):
        
        self.alpha = 0.1 # learning rate
        self.bias = None # bias
        self.theta = np.zeros((2,1)) # weights/parameters of the model
        self.iterations = 1000 # how many times to reestimate the parameters
        
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        
        m, n = X.shape # m = number of samples, n = number of features
        
        self.theta = np.zeros((n))
        self.bias = 0
        
        for i in range(self.iterations):
            
            # The predictions
            y_pred = sigmoid(self.theta.T @ X.T + self.bias)
            
            # The gradient of loss with respect to theta
            dtheta = (1/m)*np.dot(X.T, (y_pred - y))
            
            # The gradient of loss with respect to bias
            dbias = (1/m)*np.sum(y_pred - y)
            
            # Updating parameters and bias
            self.theta = self.theta - self.alpha*dtheta
            self.bias = self.bias - self.alpha*dbias
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        
        # Soft predictions, softPreds have values between 0 and 1
        softPreds = sigmoid(self.theta.T @ X.T + self.bias)
        
        # Predictions, preds have values 0 and 1 only
        preds = [1 if i > 0.5 else 0 for i in softPreds] # if y_pred >= 0.5 we get 1, if y_pred < 0.5 we get 0
        
        return np.array(preds)
        
        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        
