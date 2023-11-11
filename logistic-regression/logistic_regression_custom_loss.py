import numpy as np
from typing import List


class LogisticRegressionCustomLoss:
    
    """
    Custom implementation of logistic regression using first-order approximation with gradient descent.
    
    This class allows you to perform logistic regression with various loss functions, including:
    1. Cross-Entropy ("ce"): Standard logistic regression.
    2. Weighted Cross-Entropy ("weighted_ce"): Adds class weights to balance the loss.
    3. Focal Cross-Entropy ("focal_ce"): Introduces focal loss to focus on hard-to-classify examples.
    4. Normalized Cross-Entropy ("normalized_ce"): Normalizes the loss based on background click-through rate.
    
    Parameters:
    - learning_rate (float): The learning rate for gradient descent (default: 0.001).
    - n_iterations (int): The number of iterations for training (default: 100).
    - regularization (str): The type of regularization, either "l1" or "l2" (default: None).
    - reg_param (float): The regularization parameter (default: 0.5).
    - use_bias (bool): Whether to use a bias term (default: False).
    - objective (str): The choice of loss function ("ce," "weighted_ce," "focal_ce," or "normalized_ce").
    - alpha (float): The weighting factor for positive class samples in "weighted_ce" (default: 1.0).
    - gamma (float): The focusing parameter for "focal_ce" (default: 2.0).
    
    Sources:
    [1] 2014. Xinran He, Stuart Bowers, Joaquin Quiñonero Candela, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, and Ralf Herbrich. 
        Practical Lessons from Predicting Clicks on Ads at Facebook. Proceedings of 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining - ADKDD'14.
    [2] 2017. Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2017. Focal Loss for Dense Object Detection.
    [3] 2019. Logistic Regression From Scratch. Data Science, Machine Learning, Deep Learning. Alpha-quantum.com.
    [4] 2020. Chen Wang, Chengyuan Deng, and Shuzhen Wang. Imbalance-XGBoost: Leveraging Weighted and Focal Losses for Binary Label-Imbalanced Classification with XGBoost.

    """
    
    def __init__(
        self, learning_rate: float = 0.001, n_iterations: int = 100, regularization: str = None, 
        reg_param: float = 0.5, use_bias: bool = False, objective: str = 'ce', alpha: float = 0.25, gamma: float = 2.0):
        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.reg_param = reg_param
        self.use_bias = use_bias
        self.objective = objective
        self.alpha = alpha
        self.gamma = gamma

    def _initialize_weights(self, n_features: int):
        # Initialize weights with 0s
        self.weights =  np.zeros(n_features)
        # Initialize a bias column
        self.bias = 1.0 if self.use_bias else None
       
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _add_regularization(self, gradient: np.ndarray) -> np.ndarray:
        if self.regularization == 'l1':
            gradient -= self.reg_param * np.sign(self.weights)
        elif self.regularization == 'l2':
            gradient -= 2 * self.reg_param * self.weights
        return gradient

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionCustomLoss':
        
        if self.fit_bias:
            # Add a column of ones for the bias term
            ones = np.ones((X.shape[0], 1))
            X = np.column_stack((X, ones))

        n_samples, n_features = X.shape
        self._initialize_weights(n_features)

        for _ in range(self.n_iterations):
            
            y_logit = np.dot(X, self.weights)
            y_pred = self._sigmoid(y_logit)
        
            if self.objective == 'ce':
                gradient = -np.dot(X.T, (y - y_pred)) / n_samples
            elif self.objective == 'weighted_ce':
                weights = np.where(y == 1, self.alpha, 1)
                gradient = -np.dot(X.T, weights * (y - y_pred)) / n_samples
            elif self.objective == 'focal_ce':
                p_t = np.where(y == 1, y_pred, 1 - y_pred)
                gradient = -np.dot(X.T, (1 - p_t) ** self.gamma * (y - y_pred)) / n_samples
            elif self.objective == 'normalized_ce':
                background_ctr = np.mean(y)
                weights_ctr = np.where(y == 1, np.log(background_ctr), np.log(1-background_ctr))
                y_logit_ctr = np.where(y == 1, background_ctr * weights_ctr, (1-background_ctr) * weights_ctr)
                y_pred_ctr = self._sigmoid(y_logit_ctr)
                gradient = -np.dot(X.T, (y - (y_pred/y_pred_ctr))) / n_samples
            else:
                raise ValueError("Invalid objective parameter")

            if self.regularization:
                gradient = self._add_regularization(gradient)
                
            if self.use_bias:
                self.weights[:-1] -= self.learning_rate * gradient[:-1]
                self.bias += self.learning_rate * gradient[-1]
        
            self.weights -= self.learning_rate * gradient

        return self

    # TODOs: add decision rule (next sklearn release)
    def predict(self, X: np.ndarray) -> List[int]:
        linear_model = np.dot(X, self.weights)
        y_pred = self._sigmoid(linear_model)
        return [1 if p >= 0.5 else 0 for p in y_pred]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.use_bias:
            y_logit = np.dot(X, self.weights[:-1])
            y_logit += self.weights[-1]
        else:
            y_logit = np.dot(X, self.weights)
        return self._sigmoid(y_logit)