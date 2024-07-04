import argparse
import os
from time import time
from typing import Union, Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.special import xlogy

console = Console()

# Add random seed with numpy
np.random.seed(42)


class SGDLogisticRegression:
    """
    Class for stochastic gradient descent
    logistic regression. This class uses
    the Fisher information matrix (FIM) or
    the stochastic gradient descent (SGD) method
    to find the optimal weights and bias for
    the logistic regression problem.
    """

    def __init__(
        self,
        learning_rate=1.0,
        max_iter=10,
        method="fisher",
        batch_size=100,
        epsilon=1e-2,
        use_bias=True,
        print_report=False
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.method = method
        self.batch_size = batch_size
        self.weights = None
        self.use_bias = use_bias
        self.bias = 0
        self.loss_history = []
        self.weights_history = []
        self.print_report = print_report

    @staticmethod
    def logistic_function(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def compute_loss(y, p):
        return -np.mean(xlogy(y, p) + xlogy(1 - y, 1 - p))

    def fit(
        self,
        X,
        y,
        save_report=False,
        report_dir=None,
    ):
        if save_report and report_dir is None:
            raise ValueError(
                "If save_report is True, report_dir must be provided."
            )

        if self.use_bias:
            bias_vector = np.ones((X.shape[0], 1))
            X = np.column_stack((X, bias_vector))

        self.weights = np.zeros((X.shape[1], 1))
        y = y.reshape(-1, 1)
        training_report = []
        report = []
        start_time = time()
        stop_training = False

        for iteration in range(self.max_iter):
            if stop_training:
                break
            # shuffle the training data
            idx = np.random.permutation(X.shape[0])
            X, y = X[idx], y[idx]
            for i in range(0, X.shape[0], self.batch_size):
                X_i = X[i : i + self.batch_size]
                y_i = y[i : i + self.batch_size]
                z = X_i @ self.weights
                p_i = self.logistic_function(z)
                S_i = X_i.T @ (y_i - p_i)  # Score function
                W_i = np.diag((p_i * (1 - p_i)).ravel())
                I_i = (
                    X_i.T @ W_i @ X_i
                )  # Information matrix via Hessian

                # Fisher scoring (pseudo-inverse)
                if self.method == "fisher":
                    weights_update = (
                        self.weights
                        + (
                            np.linalg.pinv(I_i)
                            @ S_i
                        )
                        * self.learning_rate
                    )
                    update_description = (
                        "Inverse FIM @ Gradient (S)"
                    )

                    # Stop training upon convergence (only for Fisher)
                    if np.linalg.norm(weights_update - self.weights) < self.epsilon:
                        report.append(
                            "Weight delta between iterations is less than epsilon. "
                            "Stopping training."
                        )
                        # Update condition for training
                        stop_training = True
                        break
                
                # Stochastic gradient descent
                elif self.method == "sgd":
                    weights_update = (
                        self.weights
                        + self.learning_rate * S_i
                    )
                    update_description = (
                        "Gradient (S) * learning_rate"
                    )
                else:
                    raise ValueError(
                        "Invalid method. Use 'fisher' or 'sgd'."
                    )

                # Trace computation for the current observation
                report: list[Union[str, Any]] = [
                    "-------------------",
                    f"Iteration {1 + iteration}, Observation {1 + i}",
                    "X:\n"
                    + self.format_matrix_for_csv(X_i),
                    "Weights: "
                    + ",".join(
                        map(str, self.weights.ravel())
                    ),
                    "P: " + ",".join(map(str, p_i.ravel())),
                    "Y - P: "
                    + ",".join(
                        map(str, (y_i - p_i).ravel())
                    ),
                    "W:\n"
                    + self.format_matrix_for_csv(W_i),
                    "Fisher Information Matrix (I):\n"
                    + self.format_matrix_for_csv(I_i),
                    "Gradient (S): "
                    + ",".join(map(str, S_i.ravel())),
                    f"Update ({update_description}):\n"
                    f"{self.format_matrix_for_csv(weights_update - self.weights)}",
                    "Weights - Update: "
                    + ",".join(
                        map(
                            str,
                            (
                                self.weights
                                - (
                                    weights_update
                                    - self.weights
                                )
                            ).ravel(),
                        )
                    ),
                    f"Log loss: {str(self.compute_loss(y_i, p_i))}",
                    "-------------------",
                ]
                training_report.extend(report)

                 # Print each iteration step using rich console
                if self.print_report:
                    for line in report:
                        console.print(line)

                self.weights = weights_update
                self.weights_history.append(
                    self.weights.flatten()
                )

            current_loss = self.compute_loss(
                y,
                self.logistic_function(X @ self.weights),
            )
            self.loss_history.append(current_loss)

            if iteration == self.max_iter - 1:
                report.append(
                    "Maximum iterations reached without convergence."
                )
                break

        end_time = time()  # End the timer
        training_time = (
            end_time - start_time
        )  # Calculate the training time

        if save_report:
            report_path = os.path.join(report_dir, "training_report.txt")  # type: ignore
            with open(report_path, "w") as file:
                file.write("\n".join(training_report))
            console.print(
                f"Training report saved to {report_path}"
            )

        # Separate the bias from weights
        if self.use_bias:
            self.bias = self.weights[-1, 0]
            self.weights = self.weights[:-1]

        # Print the final weights and loss history using rich
        table = Table(
            title="SGD Logistic Regression Training Summary"
        )
        table.add_column(
            "Iteration",
            justify="right",
            style="cyan",
            no_wrap=True,
        )
        table.add_column(
            "Loss", justify="right", style="magenta"
        )

        for i, loss in enumerate(self.loss_history):
            table.add_row(str(i + 1), f"{loss:.6f}")

        console.print(table)

        console.print(
            Panel(
                f"Final Weights:\n{self.format_matrix_for_csv(self.weights)}",
                title="Final Weights",
            )
        )
        if self.use_bias:
            console.print(
                Panel(
                    f"Final Bias: {self.bias:.4f}",
                    title="Final Bias",
                )
            )
        console.print(
            Panel(
                f"Training Time: {training_time:.2f} seconds",
                title="Training Time",
            )
        )

    def predict_proba(self, X):
        return self.logistic_function(
            X @ self.weights + self.bias
        ).flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(
            int
        )

    @staticmethod
    def format_matrix_for_csv(matrix):
        return "\n".join(
            [
                ",".join(format(x, ".4f") for x in row)
                for row in matrix
            ]
        )


def main():
    parser = argparse.ArgumentParser(
        description="SGD Logistic Regression"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size",
    )
    parser.add_argument(
        "--n_training_points",
        type=int,
        default=100,
        help="Number of training points",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=10,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="fisher",
        help="Method to use for updating weights ('fisher' or 'sgd')",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-2,
        help="Convergence threshold",
    )
    parser.add_argument(
        "--save_report",
        action="store_true",
        help="Save training report to a file",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="Directory to save the training report",
    )
    parser.add_argument(
        "--use_bias",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use bias term",
    )
    parser.add_argument(
        "--print_report",
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help="Print training report to the console (True or False)",
    )
    args = parser.parse_args()

    console.print(
        "[bold green]Training started![/bold green]"
    )

    # Example data for experiment
    X = np.array(
        [
            [0.258537073, 93.8],
            [0.005482799, 100.0],
            [0.0, 100.0],
            [0.847040738, 93.8],
            [0.991140524, 70.0],
            [0.139903684, 100.0],
            [0.712262571, 100.0],
            [0.016017375, 92.3],
            [0.271222139, 100.0],
            [0.139975004, 95.0],
            [0.066736591, 100.0],
            [0.449005923, 100.0],
            [0.027520947, 84.6],
            [0.600056563, 100.0],
            [0.9999999, 100.0],
            [0.882055735, 100.0],
            [0.266128481, 100.0],
            [0.070793228, 100.0],
            [0.224500347, 100.0],
            [0.029268105, 100.0],
            [0.310491826, 100.0],
            [0.007414453, 100.0],
            [0.026825736, 100.0],
            [0.531819456, 100.0],
            [0.046015639, 92.3],
            [0.241108769, 100.0],
            [0.005790399, 92.9],
            [0.767990854, 100.0],
            [0.389300613, 100.0],
            [0.063448777, 100.0],
            [0.242085504, 100.0],
            [0.057643046, 100.0],
            [0.0, 97.5],
            [0.173961342, 98.0],
            [0.322863191, 93.8],
            [0.556301464, 96.2],
            [0.277150878, 100.0],
            [0.9999999, 87.5],
            [0.144782191, 100.0],
            [0.297824049, 72.2],
            [0.000185107, 100.0],
            [0.013857987, 93.3],
            [0.0479988, 100.0],
            [0.63895812, 100.0],
            [0.9999999, 90.9],
            [0.036306288, 78.6],
            [0.651068771, 92.6],
            [0.307455803, 95.7],
            [0.654946615, 100.0],
            [0.046015639, 93.3],
            [0.046265896, 100.0],
            [0.0, 89.0],
            [1.0, 85.0],
            [0.9999999, 94.1],
            [0.112688023, 92.3],
            [0.046751484, 100.0],
            [0.0, 100.0],
            [0.031407663, 77.8],
            [0.970227429, 100.0],
            [0.9999999, 87.5],
            [0.075374991, 100.0],
            [0.9999999, 100.0],
            [0.072297111, 95.5],
            [0.027350397, 88.9],
            [0.052532876, 94.1],
            [0.032910797, 82.4],
            [0.414840121, 100.0],
            [0.915582442, 92.9],
            [0.068049255, 100.0],
            [0.010399653, 81.8],
            [0.317121919, 85.7],
            [0.505825074, 100.0],
            [1.0, 97.8],
            [0.037495168, 100.0],
            [0.853409773, 97.0],
            [0.298356198, 100.0],
            [0.405018593, 97.6],
            [0.038872021, 100.0],
            [0.003607772, 100.0],
            [0.96969112, 81.3],
            [0.073448164, 57.9],
            [0.194782921, 90.5],
            [0.039903708, 94.4],
            [0.826173826, 100.0],
            [0.501213529, 100.0],
            [0.035503263, 93.8],
            [0.037915017, 100.0],
            [0.634592215, 95.5],
            [0.666212163, 100.0],
            [0.440600665, 91.7],
            [0.158253997, 87.5],
            [0.448955104, 100.0],
            [0.956583833, 100.0],
            [0.433571345, 97.8],
            [0.873258849, 100.0],
            [0.226051597, 100.0],
            [0.021293067, 100.0],
            [0.061872382, 100.0],
            [0.0809386, 84.0],
            [0.020561215, 100.0],
        ]
    )

    y = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    X_train = X[:int(args.n_training_points)]
    y_train = y[:int(args.n_training_points)]

    model = SGDLogisticRegression(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        method=args.method,
        batch_size=args.batch_size,
        epsilon=args.epsilon,
        use_bias=args.use_bias,
        print_report=args.print_report,
    )
    model.fit(
        X_train,
        y_train,
        save_report=args.save_report,
        report_dir=args.report_dir,
    )

    if args.save_report:
        console.print("[bold green]Training complete and report saved![/bold green]")


if __name__ == "__main__":
    main()

# Example usage:
# python3 sgd_logistic_regression.py --learning_rate 1.0 --method "fisher" --use_bias --max_iter 10 --batch_size 100 --n_training_points 100 --epsilon 1.0 --print_report False