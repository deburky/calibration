import numpy as np
from sklearn.metrics import log_loss
from scipy.optimize import minimize

class TemperatureScaling:
    """A class for temperature scaling calibration.
    
    Exemplary invocation:
    TS = TemperatureScaling(sample_weight=0.9040)
    TS.fit(y_train, y_pred_train)
    y_pred_calibrated = TS.predict_proba(y_pred_test)
    """
    
    def __init__(self, sample_weight=None):
        """Initialize the temperature scaling calibration class.
        
        Parameters
        ----------
        sample_weight : float, optional
            Sample weight (constant). If not provided, no sample weight is used.
        """
        self.sample_weight = sample_weight

    def fit(self, y_true, y_pred):
        """Fit the model to the training data.
        
        Parameters
        ----------
        y_true : array-like, shape (n_samples,)
            The true labels of the training data.
        y_pred : array-like, shape (n_samples,)
            The predicted probabilities of the training data.
        """
        # avoiding 0s
        if min(y_pred) == 0:
            y_pred += 1e-7
            
        if self.sample_weight is not None:
            weights = np.ones(len(y_true))
            mask = (y_true == 0)
            weights[mask] = self.sample_weight     
        else:
            weights=None

        # temperature scaling
        def scale_fun_ce(x, *args):
            t = x[0]
            y_pred_scaled = y_pred / t
            return log_loss(y_true, y_pred_scaled, sample_weight=weights)

        min_obj = minimize(
            scale_fun_ce, 
            [1],
            method='Nelder-Mead',
            options={'xatol': 1e-13, 
                     'disp': False})
        
        self.scalar_ = min_obj.x[0]
        
    def predict_proba(self, y_pred):
        if min(y_pred) == 0:
            y_pred += 1e-7 

        return y_pred / self.scalar_