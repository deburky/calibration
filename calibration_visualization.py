"""
Calibration benchmarking.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from calibration_module import CalibrationModule  # type: ignore
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Try to import various ML libraries, handle gracefully if not available
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Note: catboost not available. Install with: pip install catboost")

try:
    from xgboost import XGBClassifier, XGBRFClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: xgboost not available. Install with: pip install xgboost")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import venn_abers, handle gracefully if not available
try:
    from venn_abers import VennAbersCalibrator

    VENN_ABERS_AVAILABLE = True
except ImportError:
    VENN_ABERS_AVAILABLE = False
    print("Note: venn_abers not available. Install with: pip install venn-abers")


# Set up plotting parameters - clean and compact
plt.rcParams["font.family"] = "Avenir"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

# Set common figsize
min_x, min_y = 4, 4

# Create output directory
output_dir = Path("images")
output_dir.mkdir(exist_ok=True)


class PDCalibrationDemo:
    """PD Calibration demonstration following LGD models.ipynb approach"""

    def __init__(self, n_samples=10_000, true_pd=0.3, seed=42):
        np.random.seed(seed)
        self.n_samples = n_samples
        self.true_pd = true_pd
        self.calib = CalibrationModule()  # Following notebook approach
        self.generate_data()

    def generate_data(self, synthetic=False):
        """Generate data for the demonstration."""
        if synthetic:
            # Generate classification data with target class distribution
            self.X, self.y = make_classification(
                n_samples=self.n_samples,
                n_features=10,
                n_informative=10,
                n_redundant=0,
                n_clusters_per_class=3,
                class_sep=1,
                weights=[
                    1 - self.true_pd,
                    self.true_pd,
                ],  # Set class distribution directly
                random_state=42,
            )

            print(
                f"Generated data: {self.y.sum()} defaults out of {len(self.y)} ({self.y.mean():.3%})"
            )
        else:
            # path_to_data = (
            #     "https://raw.githubusercontent.com/deburky/boosting-scorecards/"
            #     "refs/heads/main/rfgboost/BankCaseStudyData.csv"
            # )
            path_to_data = (
                "/Users/deburky/Documents/python/python-ml-projects/"
                "random-forest/BankCaseStudyData.csv"
            )
            df = pd.read_csv(path_to_data)
            df["is_default"] = df["Final_Decision"].map({"Accept": 0, "Decline": 1})

            self.X = df[
                [
                    "Application_Score",
                    "Bureau_Score",
                    "Gross_Annual_Income",
                ]
            ].values
            self.y = df["is_default"].values

            print(
                f"Imported data: {self.y.sum()} defaults out of {len(self.y)} ({self.y.mean():.3%})"
            )

    def get_base_estimator(self, model_type="catboost"):
        """Get base estimator based on model type"""
        if model_type == "catboost" and CATBOOST_AVAILABLE:
            return CatBoostClassifier(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            )
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                objective="binary:logistic",
            )
        elif model_type == "xgbrf" and XGBOOST_AVAILABLE:
            return XGBRFClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=42,
                verbosity=0,
                objective="binary:logistic",
            )

        else:  # Default to logistic regression
            return LogisticRegression(
                fit_intercept=True, solver="newton-cg", random_state=42
            )

    def get_model_name(self, model_type):
        """Get clean model name for output files"""
        name_map = {
            "catboost": "CatBoost",
            "xgboost": "XGBoost",
            "xgbrf": "XGBRandomForest",
            "logistic": "LogisticRegression",
        }
        return name_map.get(model_type, "LogisticRegression")

    def get_raw_scores(self, model, X, model_type):
        """Get raw scores from model based on type"""
        if (
            model_type == "catboost"
            and CATBOOST_AVAILABLE
            and isinstance(model, CatBoostClassifier)
        ):
            return model.predict(X, prediction_type="RawFormulaVal")
        elif (
            model_type in ["xgboost", "xgbrf"]
            and XGBOOST_AVAILABLE
            and hasattr(model, "predict")
        ):
            if "output_margin" in model.predict.__code__.co_varnames:
                return model.predict(X, output_margin=True)
            # Fallback: convert probabilities to logits
            probs = model.predict_proba(X)[:, 1]
            return np.log(probs / (1 - probs + 1e-15))

        else:
            # For LogisticRegression, use decision_function
            return model.decision_function(X)

    def supports_sample_weights(self, model, model_type):
        """Check if model supports sample weights"""
        return model_type != "logistic"

    def create_balanced_sample(self, ratio=1.0):
        """Create balanced sample (case-control design)"""
        defaults = np.where(self.y == 1)[0]
        non_defaults = np.where(self.y == 0)[0]

        n_defaults = len(defaults)
        n_controls = int(n_defaults * ratio)

        selected_controls = np.random.choice(non_defaults, n_controls, replace=False)
        balanced_idx = np.concatenate([defaults, selected_controls])
        np.random.shuffle(balanced_idx)

        return balanced_idx

    # pylint disable: invalid-name
    def train_models(self, model_type="catboost"):
        """Train all three models and return predictions."""
        # Split full data first
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.3,
            random_state=42,
            stratify=self.y,
        )

        # Create balanced training sample
        balanced_idx = self.create_balanced_sample(ratio=1.0)
        X_train = self.X[balanced_idx]
        y_train = self.y[balanced_idx]

        # 1. Uncorrected model on original data (no rebalancing)
        model_uncorrected = self.get_base_estimator(model_type)
        model_uncorrected.fit(X_train_full, y_train_full)  # Train on original data
        pred_uncorrected = model_uncorrected.predict_proba(X_test)[:, 1]

        # 2. Weighted model (King-Zeng implementation on balanced sample)
        # pylint disable: invalid-name
        DR_SAMPLE = np.mean(y_train)  # Balanced sample rate (0.5)
        DR_TARGET = np.mean(y_train_full)  # Original PD target (0.03)

        # Calculate weights following King-Zeng (2001)
        # (1 - true_positive_rate_past) / (1 - y_train.mean()) in probabl.
        w0 = (1 - DR_TARGET) / (1 - DR_SAMPLE)  # weight for non-defaults
        # true_positive_rate_past / y_train.mean() in probabl.
        w1 = DR_TARGET / DR_SAMPLE  # weight for defaults

        # Handle weights based on model type
        model_weighted = self.get_base_estimator(model_type)

        if self.supports_sample_weights(model_weighted, model_type):
            # Use sample weights
            sample_weights = np.ones(len(y_train))
            sample_weights[y_train == 0] = w0 / w1  # Relative weights for non-defaults
            sample_weights[y_train == 1] = 1.0  # Baseline weight for defaults
            model_weighted.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            # Use class weights for LogisticRegression
            if model_type == "logistic":
                cw_dict = {0: w0 / w1, 1: 1.0}  # {0: w0, 1: w1} gives a similar result
                model_weighted = LogisticRegression(
                    fit_intercept=True,
                    class_weight=cw_dict,
                    random_state=42,
                )

            model_weighted.fit(X_train, y_train)
        pred_weighted = model_weighted.predict_proba(X_test)[:, 1]

        # 3. Prior-corrected model (King–Zeng §4.1): pure intercept shift in logit space
        model_prior_base = self.get_base_estimator(model_type)
        model_prior_base.fit(X_train, y_train)  # trained on balanced data (ȳ≈0.5)

        prior = np.log(((1 - DR_TARGET) / DR_TARGET) * (DR_SAMPLE / (1 - DR_SAMPLE)))
        # prior = logit(true_positive_rate_past) - logit(y_train.mean()) in probabl.

        if model_type == "logistic":
            model_prior_base.intercept_ -= prior
            pred_prior = model_prior_base.predict_proba(X_test)[:, 1]
        else:
            s_test = self.get_raw_scores(model_prior_base, X_test, model_type).reshape(
                -1, 1
            )
            pred_prior = expit(s_test.ravel() - prior)
            pred_prior = np.clip(pred_prior, 1e-12, 1 - 1e-12)

        # 4. Raw score recalibration approach
        # Train on full sample to get raw scores
        model_raw = self.get_base_estimator(model_type)
        model_raw.fit(X_train_full, y_train_full)

        # Get raw scores on training and test sets
        raw_logits_train = self.get_raw_scores(model_raw, X_train_full, model_type)
        raw_logits_test = self.get_raw_scores(model_raw, X_test, model_type)

        # Fit recalibration model: logistic regression on raw logits
        # This learns the mapping from raw scores to true probabilities
        recal_model = LogisticRegressionCV(fit_intercept=True, random_state=42, cv=10)
        recal_model.fit(raw_logits_train.reshape(-1, 1), y_train_full)

        # Get recalibrated predictions
        pred_recal = recal_model.predict_proba(raw_logits_test.reshape(-1, 1))[:, 1]

        # 5. Isotonic calibration (from balanced sample to target distribution)
        # Train base model on balanced sample
        base_model_isotonic = self.get_base_estimator(model_type)
        base_model_isotonic.fit(X_train, y_train)

        # Apply isotonic calibration using full training data for calibration
        isotonic_calibrator = CalibratedClassifierCV(
            base_model_isotonic,
            method="isotonic",
            cv="prefit",  # Use prefit model
        )
        isotonic_calibrator.fit(X_train_full, y_train_full)  # Calibrate on full data
        pred_isotonic = isotonic_calibrator.predict_proba(X_test)[:, 1]

        # 6. Venn-ABERS calibration (if available)
        pred_venn_abers = None
        venn_params = {}
        if VENN_ABERS_AVAILABLE:
            try:
                # Train base model on balanced sample
                base_model_va = self.get_base_estimator(model_type)

                # Create Venn-ABERS calibrator
                va_calibrator = VennAbersCalibrator(
                    estimator=base_model_va,
                    inductive=True,
                    cal_size=0.3,
                    random_state=42,
                )

                # Fit on full training data (it will handle the calibration split)
                va_calibrator.fit(X_train_full, y_train_full)

                # Get predictions
                pred_venn_abers = va_calibrator.predict_proba(X_test)[:, 1]
                venn_params = {"venn_abers_available": True}
            except ImportError as e:
                print(f"Venn-ABERS calibration failed: {e}")
                venn_params = {"venn_abers_available": False}
        else:
            venn_params = {"venn_abers_available": False}

        # Prepare results dictionary
        predictions = {
            "uncorrected": pred_uncorrected,
            "weighted": pred_weighted,
            "prior": pred_prior,
            "logistic": pred_recal,
            "isotonic": pred_isotonic,
        }

        if pred_venn_abers is not None:
            predictions["venn_abers"] = pred_venn_abers

        return {
            "X_test": X_test,
            "y_test": y_test,
            "predictions": predictions,
            "params": {
                "sample_dr": DR_SAMPLE,  # Balanced sample DR
                "target_dr": DR_TARGET,  # Original DR target
                "original_dr": np.mean(y_train_full),  # Original full sample DR
                "w0": w0,
                "w1": w1,
                "prior": prior,
                "recal_intercept": recal_model.intercept_[0],
                "recal_slope": recal_model.coef_[0][0],
                **venn_params,
            },
        }

    def plot_all_calibration_curves(self, results, model_name="Model"):
        # sourcery skip: class-extract-method
        """Plot all calibration curves in a grid layout"""
        n_methods = len(results["predictions"])

        # Determine grid layout (prefer 3 columns)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(min_x * n_cols, min_y * n_rows), dpi=500
        )
        fig.suptitle(
            f"{model_name} - Calibration Curves", fontsize=16, fontweight="bold", y=1.01
        )
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_methods > 1 else [axes]
        else:
            axes = axes.flatten()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        colors = ["#c430c1", "#ffa94d", "#55d3ed", "#69db7c", "#ff6b6b", "#4ecdc4"]

        # Plot each model's calibration curve
        for i, (name, preds) in enumerate(results["predictions"].items()):
            # Data preparation
            preds_df = pd.DataFrame(
                {
                    "prediction": preds,
                    "label": results["y_test"],
                }
            )

            # Calculate metrics
            brier = brier_score_loss(preds_df["label"], preds_df["prediction"])
            log_loss_score = log_loss(preds_df["label"], preds_df["prediction"])

            # Plot on appropriate axis
            ax = axes[i]
            CalibrationDisplay.from_predictions(
                preds_df["label"],
                preds_df["prediction"],
                n_bins=10,
                marker="o",
                ax=ax,
                color=colors[i % len(colors)],
                name=f"Brier: {brier:.4f}\nLog Loss: {log_loss_score:.4f}",
            )

            ax.set_title(f"{name.replace('_', ' ').title()} Model", fontsize=12)
            ax.grid(True, linestyle="dotted", linewidth=0.7, alpha=0.6)
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_yticks(np.arange(0, 1.1, 0.2))

        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle
        plt.savefig(
            output_dir / f"{model_name}_calibration_curves.png",
            bbox_inches="tight",
            dpi=500,
        )
        plt.close()

    def plot_all_ccp_curves(self, results, model_name="Model"):
        """Plot Cumulative Calibration Profile curves for all models in grid."""
        n_methods = len(results["predictions"])

        # Determine grid layout (prefer 3 columns)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(min_x * n_cols, min_y * n_rows), dpi=500
        )
        fig.suptitle(
            f"{model_name} - CCP Curves", fontsize=16, fontweight="bold", y=1.01
        )
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_methods > 1 else [axes]
        else:
            axes = axes.flatten()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        colors = ["#c430c1", "#ffa94d", "#55d3ed", "#69db7c", "#ff6b6b", "#4ecdc4"]

        for i, (name, preds) in enumerate(results["predictions"].items()):
            # Data preparation
            preds_df = pd.DataFrame(
                {
                    "prediction": preds,
                    "label": results["y_test"],
                }
            ).sort_values(by="prediction", ascending=True)

            # Calculate cumulative values
            cumulative_bad = preds_df["label"].cumsum() / preds_df["label"].sum()

            # Calculate calibration error
            error = np.mean(np.abs(cumulative_bad - preds_df["prediction"]))

            # Plot on appropriate axis
            ax = axes[i]
            ax.plot(
                cumulative_bad,
                preds_df["prediction"],
                label=f"CE: {error:.2%}",
                color=colors[i % len(colors)],
            )

            # Format plot
            ax.set_title(f"{name.replace('_', ' ').title()} Model", fontsize=12)
            ax.set_xlabel("Fraction of bads")
            ax.set_ylabel("Predicted probability")
            ax.grid(True, linestyle="dotted", linewidth=0.7, alpha=0.6)
            ax.plot([0, 1], [0, 1], linestyle="dotted", color="black", alpha=0.5)
            ax.legend(loc="upper left")
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_yticks(np.arange(0, 1.1, 0.2))

        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle
        plt.savefig(
            output_dir / f"{model_name}_ccp_curves.png",
            bbox_inches="tight",
            dpi=500,
        )
        plt.close()

    def plot_all_reliability_diagrams(self, results, model_name="Model"):
        """Plot reliability diagrams for all models in grid."""
        n_methods = len(results["predictions"])

        # Determine grid layout (prefer 3 columns)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(min_x * n_cols, min_y * n_rows), dpi=500
        )
        fig.suptitle(
            f"{model_name} - Reliability Diagrams",
            fontsize=16,
            fontweight="bold",
            y=1.01,
        )
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_methods > 1 else [axes]
        else:
            axes = axes.flatten()

        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        for i, (name, preds) in enumerate(results["predictions"].items()):
            # Calculate bins and metrics
            bins, _, prob_true, _, _ = self.calib.calc_bins(results["y_test"], preds)
            ece, mce = self.calib.calc_metrics(results["y_test"], preds)

            # Plot on appropriate axis
            ax = axes[i]

            # Plot components
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.grid(True, linestyle="dotted", linewidth=0.7, alpha=0.6)
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

            # Draw bars
            ax.bar(
                bins,
                bins,
                width=0.1,
                alpha=0.3,
                edgecolor="red",
                color="red",
                hatch="\\",
                label="Gap",
            )
            ax.bar(
                bins,
                prob_true,
                width=0.1,
                alpha=1,
                edgecolor="black",
                color="blue",
                label="Outputs",
            )

            # Add legends
            ax.legend(loc="upper right", fontsize=8)
            ax.text(
                0.05,
                0.95,
                f"ECE = {ece:.2%}\nMCE = {mce:.2%}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )

            ax.set_title(f"{name.replace('_', ' ').title()} Model", fontsize=12)

        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle
        plt.savefig(
            output_dir / f"{model_name}_reliability_diagrams.png",
            bbox_inches="tight",
            dpi=500,
        )
        plt.close()


def run_single_model_demo(demo, model_type):
    """Run demo for a single model type."""
    model_name = demo.get_model_name(model_type)
    print(f"\n{'=' * 70}")
    print(f"Running {model_name} Analysis")
    print(f"{'=' * 70}")

    # Train all models
    print(f"\n1. Training {model_name} models...")
    results = demo.train_models(model_type=model_type)

    # Plot all visualizations
    print(f"\n2. Generating {model_name} visualizations...")

    # Plot standard calibration curves
    print("  - Plotting calibration curves...")
    demo.plot_all_calibration_curves(results, model_name)

    # Plot CCP curves
    print("  - Plotting CCP curves...")
    demo.plot_all_ccp_curves(results, model_name)

    # Plot reliability diagrams
    print("  - Plotting reliability diagrams...")
    demo.plot_all_reliability_diagrams(results, model_name)

    # Print results summary
    print(f"\n3. {model_name} Results Summary:")
    print("=" * 70)
    print(f"True Population DR: {demo.true_pd:.3%}")
    print(f"Balanced Sample DR: {results['params']['sample_dr']:.3%}")
    print(f"Target DR: {results['params']['target_dr']:.3%}")
    print("\nPrediction Distributions:")

    for name, preds in results["predictions"].items():
        print(f"\n{name.title()} Model:")
        print("  Distribution Statistics:")
        print(f"    Mean:   {preds.mean():.3%}")
        print(f"    Std:    {preds.std():.3%}")
        print(f"    Min:    {preds.min():.3%}")
        print(f"    25th:   {np.percentile(preds, 25):.3%}")
        print(f"    Median: {np.percentile(preds, 50):.3%}")
        print(f"    75th:   {np.percentile(preds, 75):.3%}")
        print(f"    Max:    {preds.max():.3%}")
        print("  Performance Metrics:")
        auc = roc_auc_score(results["y_test"], preds)
        brier = brier_score_loss(results["y_test"], preds)
        ece, mce = demo.calib.calc_metrics(results["y_test"], preds)
        print(f"    AUC:   {auc:.3f}")
        print(f"    Brier: {brier:.4f}")
        print(f"    ECE:   {ece:.3%}")
        print(f"    MCE:   {mce:.3%}")

    print("\nCalibration Parameters:")
    print(f"  Original DR:       {results['params']['original_dr']:.3%}")
    print(f"  Balanced Sample DR:{results['params']['sample_dr']:.3%}")
    print(f"  Target DR:         {results['params']['target_dr']:.3%}")
    print(f"  w0 (non-defaults): {results['params']['w0']:.4f}")
    print(f"  w1 (defaults):     {results['params']['w1']:.4f}")
    print(
        f"  Relative weight:   {results['params']['w0'] / results['params']['w1']:.4f}"
    )
    print(f"  Prior correction:  {results['params']['prior']:.4f}")
    print(f"  Recal intercept:   {results['params']['recal_intercept']:.4f}")
    print(f"  Recal slope:       {results['params']['recal_slope']:.4f}")

    return results, model_name


def main():
    """Run the complete demonstration for all model types."""
    print("PD Calibration Weights Demonstration")
    print("Comparing Multiple Base Estimators")
    print("=" * 70)

    # Initialize demo (using same data for all models)
    demo = PDCalibrationDemo(n_samples=1000, true_pd=0.03)

    # Define model types to test
    model_types = ["logistic", "catboost", "xgboost", "xgbrf"]

    # Store all results for comparison
    all_results = {}

    for model_type in model_types:
        try:
            results, model_name = run_single_model_demo(demo, model_type)
            all_results[model_name] = results
        except (ValueError, ImportError) as e:
            print(f"Error running {model_type}: {e}")
            continue

    # Final output message
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"All plots saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for model_name in all_results:
        print(f"  {model_name}_calibration_curves_combined.png")
        print(f"  {model_name}_ccp_curves_combined.png")
        print(f"  {model_name}_reliability_diagrams_combined.png")


if __name__ == "__main__":
    main()
