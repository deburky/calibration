import numpy as np
import matplotlib.pyplot as plt

class CalibrationModule():
    
    def __init__(self, n_bins=10, min_proba=0.1, max_proba=1):
        """ Define parameters for binning.
        
        Parameters
        ----------
        n_bins : int, default=10
            Number of bins
        min_proba : float, default=0.1
            Probability range minimum for bins
        max_proba : float, default=1
            Probability range maximum for bins
        """
        
        self.n_bins = n_bins
        self.min_proba = min_proba
        self.max_proba = max_proba
        
    def calc_bins(self, y_true, y_pred):
        """Calculate bins with observed and predicted values.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted probabilities
        """
        
        # Define bin ranges
        bins = np.linspace(self.min_proba, self.max_proba, self.n_bins)
        # Return the indices of the bins to which each value in input array belongs
        binned_preds = np.digitize(y_pred, bins)

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
        return bins, binned_preds, bin_true, bin_pred, bin_size
    
    def calc_metrics(self, y_true, y_pred):
        """ Calculate ECE and MCE metrics.
        
        Expected Calibration Error (ECE) is a measure of the average difference between 
        the observed accuracy and the predicted confidence for a classifier, while Maximum 
        Calibration Error (MCE) is the maximum difference between these quantities.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted probabilities    
        """        
        
        ece = 0
        mce = 0
        bins, binned_preds, bin_true, bin_pred, bin_size = self.calc_bins(y_true, y_pred)

        for bin_idx, _ in enumerate(bins):
            abs_difference = abs(bin_true[bin_idx] - bin_pred[bin_idx])
            ece += (bin_size[bin_idx] / sum(bin_size)) * abs_difference
            mce = max(mce, abs_difference)

        return round(ece, 4), round(mce, 4)
    
    def plot_reliability_diagram(self, y_true, y_pred):
        """Draw reliability diagram.

        The reliability diagram is a graphical tool for visualizing the calibration 
        of a classifier. It plots the predicted probabilities (or confidences) of 
        the positive class on the x-axis and the observed accuracies on the y-axis, 
        and it displays the number of test examples in each bin formed by dividing 
        the range of predicted probabilities into equal-sized intervals.
        This code calculates the predicted probabilities, observed accuracies, and bin 
        sizes for each interval, and it plots them as a bar chart on a scatter plot, 
        with error bars indicating the uncertainty in the observed accuracies.

        In addition to the reliability diagram, this code also calculates and displays 
        the Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) 
        for the classifier, as defined in Guo et al 2017.

        Note (Guo et al. 2017):
        ... reliability diagrams do not display the proportion 
        of samples in a given bin, and thus cannot be used to estimate 
        how many samples are calibrated.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted probabilities    

        """

        bins, binned_preds, prob_true, prob_pred, bin_size = self.calc_bins(y_true, y_pred)
        ece, mce = self.calc_metrics(y_true, y_pred)

        fig = plt.figure(figsize=(8, 8), dpi=600)
        ax = fig.gca()

        # x/y limits
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)

        # x/y labels
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')

        # Create grid
        ax.set_axisbelow(True)
        ax.grid(color='gray', linestyle='dashed')

        # Draw identity line
        plt.plot([0, 1],[0, 1], '--', color='gray', linewidth=2)
        # Draw gap bars
        plt.bar(bins, bins, width=0.1, alpha=0.3, edgecolor='red', color='red', hatch='\\', label='Gap')
        # Draw output bars
        plt.bar(bins, prob_true, width=0.1, alpha=1, edgecolor='black', color='b', label='Outputs')

        # Equally spaced axes
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Add legend for gap and output bars
        gap_output_legend = plt.legend(loc='upper left')     

        # ECE and MCE legend
        ece_patch = mpatches.Patch(color='green', label=f'ECE = {ece:.2%}')
        mce_patch = mpatches.Patch(color='red', label=f'MCE = {mce:.2%}')
        
        ece_mce_legend = plt.legend(handles=[ece_patch, mce_patch], loc='lower right')
        
        # Add both legends to the chart
        ax.add_artist(gap_output_legend)
        ax.add_artist(ece_mce_legend)
        plt.tight_layout()

        plt.show()
        
    def plot_calibration_curve(self, y_true, y_pred, ax=None):
        """ Draw a calibration curve.
        
        This function 

        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted probabilities
        """
        
        bins, binned_preds, prob_true, prob_pred, bin_size = self.calc_bins(y_true, y_pred)
        ece, mce = self.calc_metrics(y_true, y_pred)
        
        if ax==None:
            fig = plt.figure(figsize=(8, 8), dpi=600)
            ax = plt.gca()
        else:
            plt.sca(ax)
        plt.plot([0, 1],[0, 1], '--', color='gray', linewidth=2, label="Perfectly calibrated model")
        plt.plot(prob_true, prob_pred, "s-", color='#7570b3', label = 'Classifier')
        
        # x/y labels
        plt.ylabel("Actual probability")
        plt.xlabel("Predicted probability")

        plt.legend()
        plt.xticks()
        plt.yticks()

        plt.grid(color='gray', linestyle='dashed')

        plt.tight_layout()

        plt.show()
