# Perceptron
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=100, batch_size=1):
        """
        Initializes a Perceptron object.

        Args:
            learning_rate (float): The learning rate for the perceptron. Default is 0.01.
            n_iter (int): The number of iterations. Default is 100.
            batch_size (int): The number of samples per batch. Default is 1.

        Returns:
            None
        """
        self.is_fitted_ = False
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.activation = self._unit_step_func
        self.weights = None
        self.bias = None
        self.weight_history = []
        self.loss_history = []
        
    def fit(self, X, y):
        """
        Fits the Perceptron model to the training data.

        Args:
            X (array-like): The input samples.
            y (array-like): The target values.

        Returns:
            Perceptron: The fitted Perceptron model.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.weight_history = []
        self.loss_history = []
        self.eval_metrics = []

        y_ = np.array([1 if i > 0 else 0 for i in y])

        # Stochastic Gradient Descent
        for _ in range(self.n_iter):
            for idx in range(0, n_samples, self.batch_size):
                end_idx = min(idx + self.batch_size, n_samples)
                X_batch = X[idx:end_idx]
                y_batch = y_[idx:end_idx]

                y_logit = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.activation(y_logit)
                
                errors_batch = y_batch - y_pred
                update = self.learning_rate * errors_batch

                # Perform averaging of the updates for each weight across the mini-batch
                weight_updates = np.dot(X_batch.T, update) / X_batch.shape[0]
                bias_update = update.mean()

                self.weights += weight_updates
                self.bias += bias_update
            
            # Log Loss (Binary Cross-Entropy Loss)
            y_logit = np.dot(X, self.weights) + self.bias
            y_pred = self.activation(y_logit)
            y_pred_proba = np.where(y_pred == 1, 0.999, 1 - 0.999)
            log_loss_score = -np.mean(y_ * np.log(y_pred_proba) + (1 - y_) * np.log(1 - y_pred_proba))
            
            # Classification error
            accuracy = accuracy_score(y_, y_pred)
            self.eval_metrics.append(1 - accuracy)

            self.loss_history.append(log_loss_score)
            self.weight_history.append((self.weights.copy(), self.bias))
        self.is_fitted_ = True  # Set is_fitted_ to True after fitting
        return self
  
    def predict(self, X):
        """
        Predicts the target values for the input samples.

        Args:
            X (array-like): The input samples.

        Returns:
            array-like: The predicted target values.
        """
        if not self.is_fitted_:
            raise NotFittedError(
                "This Classifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        y_pred = np.dot(X, self.weights) + self.bias
        return self.activation(y_pred)

    def _unit_step_func(self, x):
        """
        Defines the unit step function for the Perceptron.

        Args:
            x (array-like): The input values.

        Returns:
            array-like: The output values after applying the unit step function.
        """
        return np.where(x >= 0, 1, 0)