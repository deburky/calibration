import numpy as np

class HistogramBinner:
    """
    =============================================
    Histogram binning for calibration
    =============================================

    We discretize predicted probabilities into a set of bins of equal size. 
    For each bin we compute lower and upper boundary and calculate an average.
    We then estimate the corrected probability based on actually observed
    ground truth label (accuracy) in a corresponding bin.

    Read more about histogram method in Zadrozny & Elkan, 2001.

    Examples
    ----------
    BinClassifier = HistogramBinner()
    # learn calibration map
    y_pred_train = clf.predict_proba(X_train)[:,1]
    BinClassifier.learn_calibration_map(y_train, y_pred_train)
    # apply calibration map
    y_pred_test = clf.predict_proba(X_test)[:,1]
    y_pred_test_calib = BinClassifier.apply_calibration_map(y_pred_test)

    References
    ----------
    Obtaining calibrated probability estimates from decision trees
    and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001
    """
    def __init__(self, n_bins=10, min_proba=0.1, max_proba=1):        
        self.n_bins = n_bins
        self.min_proba = min_proba
        self.max_proba = max_proba
        self.bins = np.linspace(min_proba, max_proba, self.n_bins)
        self.bin_true = bin_true
        self.bin_pred = bin_pred
        self.histogram_bins = []
        self.accuracy_bins = []
        
    def calculate_bins(self, y_true, y_pred):
        # Return the indices of the bins to which each value in input array belongs
        binned_preds = np.digitize(y_pred, self.bins)
        # Create empty arrays for observed and predicted bins, including bin sizes
        bin_true = np.zeros(self.n_bins)
        bin_pred = np.zeros(self.n_bins)
        bin_size = np.zeros(self.n_bins)
        # Binning
        for bin_idx in range(self.n_bins):
            bin_size[bin_idx] = len(y_pred[binned_preds == bin_idx])
            if bin_size[bin_idx] > 0:
                bin_true[bin_idx] = (y_true[binned_preds == bin_idx]).sum() / bin_size[bin_idx]
                bin_pred[bin_idx] = (y_pred[binned_preds == bin_idx]).sum() / bin_size[bin_idx]
        return bin_true, bin_pred
    
    def learn_calibration_map(self, y_true, y_pred):
        # Histogram Binning
        self.histogram_bins = []
        for i in range(len(self.bin_pred) - 1):
            last_prob = self.bin_pred[i]
            next_first_prob = self.bin_pred[i+1]
            # avoid zeros to be counted as bin edges
            if next_first_prob != 0:
                self.histogram_bins.append((last_prob + next_first_prob) / 2.0)
            else:
                # alternatively we run into situations where predictions
                # out of histogram bin range will be mapped to 0
                # as in: self.histogram_bins.append(self.histogram_bins[-1:][0])
                self.histogram_bins.append(1.0)
        # add last bin edge
        self.histogram_bins.append(1.0)
        self.accuracy_bins = np.array([np.mean(value) for value in self.bin_true])
        
    def apply_calibration_map(self, y_pred: np.ndarray) -> np.ndarray:
        # to account for edge cases where probability lies outside bin range
        eps = np.finfo(y_pred.dtype).eps
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # perform mapping
        indices = np.digitize(y_pred, self.histogram_bins)
        return self.accuracy_bins[indices]