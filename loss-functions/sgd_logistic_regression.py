import os
import argparse
import numpy as np
from scipy.special import xlogy
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from time import time

console = Console()

class SGDLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, method="fisher", epsilon=1e-3):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.method = method
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.weights_history = []
        self.bias_history = []

    @staticmethod
    def logistic_function(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def compute_loss(y, p):
        return -np.mean(xlogy(y, p) + xlogy(1 - y, 1 - p))

    def fit(self, X, y, fit_bias=False, save_report=False, report_dir=None):
        if save_report and report_dir is None:
            raise ValueError("If save_report is True, report_dir must be provided.")

        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0 if fit_bias else None  # Initialize bias if fit_bias is True
        y = y.reshape(-1, 1)
        training_report = []
        start_time = time()  # Start the timer

        for iteration in range(self.max_iter):
            for i in range(X.shape[0]):
                X_i = X[i:i+1]
                y_i = y[i:i+1]

                linear_output = X_i @ self.weights + (self.bias if fit_bias else 0)
                p_i = self.logistic_function(linear_output)
                W_i = np.diag((p_i * (1 - p_i)).ravel())
                I_i = X_i.T @ W_i @ X_i
                U_i = X_i.T @ (y_i - p_i)

                if self.method == "fisher":
                    weights_update = (
                        self.weights
                        + self.learning_rate
                        * np.linalg.inv(I_i + np.eye(I_i.shape[0]) * self.epsilon) @ U_i
                    )
                    bias_update = self.bias + self.learning_rate * np.sum(y_i - p_i) if fit_bias else None
                    update_description = "FIM @ U * learning_rate"
                elif self.method == "sgd":
                    weights_update = self.weights + self.learning_rate * U_i
                    bias_update = self.bias + self.learning_rate * np.sum(y_i - p_i) if fit_bias else None
                    update_description = "Gradient (U) * learning_rate"
                else:
                    raise ValueError("Invalid method. Use 'fisher' or 'sgd'.")

                # Trace computation for the current observation
                report = [
                    "-------------------",
                    f"Iteration {1 + iteration}, Observation {1 + i}",
                    "X:\n" + self.format_matrix_for_csv(X_i),
                    "Weights: " + ",".join(map(str, self.weights.ravel())),
                    "P: " + ",".join(map(str, p_i.ravel())),
                    "Y - P: " + ",".join(map(str, (y_i - p_i).ravel())),
                    "W:\n" + self.format_matrix_for_csv(W_i),
                    "Fisher Information Matrix (I):\n" + self.format_matrix_for_csv(I_i),
                    "Gradient (U): " + ",".join(map(str, U_i.ravel())),
                    f"Update ({update_description}):\n{self.format_matrix_for_csv(weights_update - self.weights)}",
                    "Weights - Update: " + ",".join(map(str, (self.weights - (weights_update - self.weights)).ravel())),
                    f"Log loss: {str(self.compute_loss(y_i, p_i))}",
                    "-------------------",
                ]
                if fit_bias:
                    report.insert(4, f"Bias: {str(self.bias)}")
                training_report.extend(report)

                # Print each iteration step using rich console
                for line in report:
                    console.print(line)

                self.weights = weights_update
                if fit_bias:
                    self.bias = bias_update
                self.weights_history.append(self.weights.flatten())
                if fit_bias:
                    self.bias_history.append(self.bias)

                if np.linalg.norm(weights_update - self.weights) < self.epsilon:
                    report.append(f"Convergence reached at iteration {1+iteration}, observation {1+i}")
                    # break

            current_loss = self.compute_loss(y, self.logistic_function(X @ self.weights + (self.bias if fit_bias else 0)))
            self.loss_history.append(current_loss)

            if iteration == self.max_iter - 1:
                report.append("Maximum iterations reached without convergence.")
                break

        end_time = time()  # End the timer
        training_time = end_time - start_time  # Calculate the training time

        if save_report:
            report_path = os.path.join(report_dir, "training_report.txt")  # type: ignore
            with open(report_path, "w") as file:
                file.write("\n".join(training_report))
            console.print(f"Training report saved to {report_path}")

        # Print the final weights and loss history using rich
        table = Table(title="SGD Logistic Regression Training Summary")
        table.add_column("Iteration", justify="right", style="cyan", no_wrap=True)
        table.add_column("Loss", justify="right", style="magenta")

        for i, loss in enumerate(self.loss_history):
            table.add_row(str(i + 1), f"{loss:.6f}")

        console.print(table)

        console.print(Panel(f"Final Weights:\n{self.format_matrix_for_csv(self.weights)}", title="Final Weights"))
        if fit_bias:
            console.print(Panel(f"Final Bias: {self.bias}", title="Final Bias"))
        console.print(Panel(f"Training Time: {training_time:.2f} seconds", title="Training Time"))

    def predict_proba(self, X):
        return self.logistic_function(X @ self.weights + (self.bias if self.bias is not None else 0)).flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

    @staticmethod
    def format_matrix_for_csv(matrix):
        return "\n".join([",".join(map(str, row)) for row in matrix])


def main():
    parser = argparse.ArgumentParser(description="SGD Logistic Regression")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for SGD")
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--method", type=str, default="fisher", help="Method to use for updating weights ('fisher' or 'sgd')")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Convergence threshold")
    parser.add_argument("--save_report", action="store_true", help="Save training report to a file")
    parser.add_argument("--report_dir", type=str, default=None, help="Directory to save the training report")
    parser.add_argument("--fit_bias", action="store_true", help="Fit bias term")
    args = parser.parse_args()

    # Example data
    # X = np.array([
    #     [0.0, 50, 69, 30, 64, 14132.9],
    #     [0.847040738, 93.8, 97.0, 18.0, 41.0, 22897.1],
    #     [0.991140524, 70.0, 91.0, 30.0, 15.0, 2798.11],
    #     [0.307455803, 95.7, 22.0, 16.0, 73.0, 20768.85],
    #     [0.310491826, 100.0, 23.0, 8.0, 99.0, 27050.74],
    # ])
    
    # y = np.array(
    #     [
    #         # 0,
    #         # 0,
    #         1,
    #         1,
    #         # 0
    #     ]
    # )
    
    X = np.array(
        [
            [0.258537073, 93.8, 3.0, 12.0, 73.0, 26447.01],
            [0.005482799, 100.0, 69.0, 30.0, 64.0, 14132.9],
            [0.0, 100.0, 20.0, 9.0, 104.0, 17147.14],
            [0.847040738, 93.8, 97.0, 18.0, 41.0, 22897.1],
            [0.991140524, 70.0, 91.0, 30.0, 15.0, 2798.11],
            [0.139903684, 100.0, 11.0, 15.0, 140.0, 2621.67],
            [0.712262571, 100.0, 69.0, 9.0, 118.0, 25439.26],
            [0.016017375, 92.3, -9.0, 15.0, -9.0, 18701.53],
            [0.271222139, 100.0, 3.0, 15.0, 135.0, 3649.12],
            [0.139975004, 95.0, 75.0, 13.0, 50.0, 19940.62],
            [0.066736591, 100.0, 0.0, 8.0, 146.0, 25507.11],
            [0.449005923, 100.0, 47.0, 18.0, 126.0, 19724.94],
            [0.027520947, 84.6, 30.0, 9.0, 74.0, 0.0],
            [0.600056563, 100.0, 41.0, 10.0, 70.0, 23119.84],
            [0.9999999, 100.0, 7.0, 4.0, 75.0, 13185.1],
            [0.882055735, 100.0, 3.0, 17.0, 95.0, 30511.2],
            [0.266128481, 100.0, 9.0, 15.0, 87.0, 5577.82],
            [0.070793228, 100.0, 9.0, 7.0, 46.0, 5552.72],
            [0.224500347, 100.0, 11.0, 5.0, 138.0, 9740.1],
            [0.029268105, 100.0, 10.0, 20.0, 75.0, 33537.3],
            [0.310491826, 100.0, 23.0, 8.0, 99.0, 27050.74],
            [0.007414453, 100.0, 14.0, 5.0, 54.0, 8798.08],
            [0.026825736, 100.0, 6.0, 8.0, 97.0, 11603.55],
            [0.531819456, 100.0, 34.0, 13.0, 94.0, 9247.93],
            [0.046015639, 92.3, 37.0, 10.0, 68.0, 13198.89],
            [0.241108769, 100.0, 1.0, 30.0, 60.0, 35726.63],
            [0.005790399, 92.9, 75.0, 9.0, 50.0, 19926.0],
            [0.767990854, 100.0, 6.0, 19.0, 77.0, 4497.14],
            [0.389300613, 100.0, 21.0, 20.0, 127.0, 9022.85],
            [0.063448777, 100.0, 5.0, 20.0, 116.0, 6542.57],
            [0.242085504, 100.0, -9.0, 6.0, -9.0, 6247.34],
            [0.057643046, 100.0, 61.0, 14.0, 67.0, 14986.61],
            [0.0, 97.5, 12.0, 30.0, 65.0, 21115.47],
            [0.173961342, 98.0, 1.0, 37.0, 94.0, 17988.61],
            [0.322863191, 93.8, 3.0, 8.0, 90.0, 24670.99],
            [0.556301464, 96.2, 6.0, 15.0, 113.0, 16154.59],
            [0.277150878, 100.0, 35.0, 13.0, 64.0, 8985.52],
            [0.9999999, 87.5, 26.0, 4.0, 91.0, 9695.43],
            [0.144782191, 100.0, 11.0, 17.0, 94.0, 7480.6],
            [0.297824049, 72.2, 2.0, 9.0, 49.0, 8151.53],
            [0.000185107, 100.0, 21.0, 18.0, 88.0, 24305.6],
            [0.013857987, 93.3, -9.0, 8.0, -9.0, 28568.14],
            [0.0479988, 100.0, 0.0, 11.0, 55.0, 8782.01],
            [0.63895812, 100.0, 66.0, 19.0, 26.0, 14223.67],
            [0.9999999, 90.9, 1.0, 6.0, 60.0, 0.0],
            [0.036306288, 78.6, 3.0, 14.0, 28.0, 19423.87],
            [0.651068771, 92.6, 3.0, 21.0, 102.0, 24350.26],
            [0.307455803, 95.7, 22.0, 16.0, 73.0, 20768.85],
            [0.654946615, 100.0, 33.0, 30.0, 103.0, 11688.12],
            [0.046015639, 93.3, 4.0, 12.0, 67.0, 28533.66],
            [0.046265896, 100.0, 3.0, 9.0, 137.0, 6708.04],
            [0.0, 89.0, 39.0, 7.0, 63.0, 23776.49],
            [1.0, 85.0, 15.0, 9.0, 67.0, 13618.79],
            [0.9999999, 94.1, 36.0, 9.0, 129.0, 33221.93],
            [0.112688023, 92.3, 57.0, 6.0, 87.0, 28000.0],
            [0.046751484, 100.0, 22.0, 11.0, 52.0, 35095.78],
            [0.0, 100.0, 28.0, 4.0, 83.0, 14236.2],
            [0.031407663, 77.8, 18.0, 12.0, 82.0, 13262.0],
            [0.970227429, 100.0, 0.0, 8.0, 29.0, 8345.75],
            [0.9999999, 87.5, 53.0, 14.0, 81.0, 4800.56],
            [0.075374991, 100.0, 0.0, 2.0, 58.0, 9135.4],
            [0.9999999, 100.0, 18.0, 25.0, 89.0, 8997.38],
            [0.072297111, 95.5, 20.0, 16.0, 73.0, 13944.54],
            [0.027350397, 88.9, -9.0, 8.0, -9.0, 2376.16],
            [0.052532876, 94.1, 14.0, 11.0, 61.0, 15928.45],
            [0.032910797, 82.4, 1.0, 8.0, 193.0, 32866.83],
            [0.414840121, 100.0, 3.0, 3.0, 80.0, 23253.54],
            [0.915582442, 92.9, 3.0, 31.0, 57.0, 4449.59],
            [0.068049255, 100.0, 5.0, 12.0, 45.0, 4675.37],
            [0.010399653, 81.8, 10.0, 19.0, 38.0, 10843.53],
            [0.317121919, 85.7, 52.0, 6.0, 60.0, 26248.35],
            [0.505825074, 100.0, 6.0, 15.0, 69.0, 8961.95],
            [1.0, 97.8, 40.0, 4.0, 138.0, 24218.29],
            [0.037495168, 100.0, 0.0, 10.0, 82.0, 35943.71],
            [0.853409773, 97.0, 53.0, 10.0, 119.0, 37929.84],
            [0.298356198, 100.0, 0.0, 17.0, 87.0, 14177.42],
            [0.405018593, 97.6, 19.0, 19.0, 65.0, 6379.88],
            [0.038872021, 100.0, 36.0, 10.0, 103.0, 16771.43],
            [0.003607772, 100.0, 77.0, 7.0, 96.0, 7925.39],
            [0.96969112, 81.3, 55.0, 8.0, 56.0, 14518.41],
            [0.073448164, 57.9, -9.0, 6.0, -9.0, 12374.37],
            [0.194782921, 90.5, 30.0, 14.0, 140.0, 17408.96],
            [0.039903708, 94.4, 3.0, 8.0, 59.0, 28408.62],
            [0.826173826, 100.0, -9.0, 8.0, -9.0, 37540.29],
            [0.501213529, 100.0, 2.0, 7.0, 114.0, 4518.12],
            [0.035503263, 93.8, 35.0, 11.0, 69.0, 8145.48],
            [0.037915017, 100.0, 14.0, 4.0, 61.0, 4528.55],
            [0.634592215, 95.5, 59.0, 10.0, 98.0, 14499.3],
            [0.666212163, 100.0, 25.0, 15.0, 83.0, 33635.31],
            [0.440600665, 91.7, 5.0, 26.0, 98.0, 0.0],
            [0.158253997, 87.5, 57.0, 10.0, 107.0, 2206.45],
            [0.448955104, 100.0, 3.0, 22.0, 59.0, 9018.44],
            [0.956583833, 100.0, 24.0, 13.0, 86.0, 6672.11],
            [0.433571345, 97.8, 3.0, 10.0, 128.0, 11140.42],
            [0.873258849, 100.0, 9.0, 17.0, 72.0, 12706.94],
            [0.226051597, 100.0, 8.0, 40.0, 75.0, 13464.86],
            [0.021293067, 100.0, 25.0, 3.0, 114.0, 7052.52],
            [0.061872382, 100.0, 10.0, 6.0, 69.0, 19005.39],
            [0.0809386, 84.0, 37.0, 14.0, 128.0, 8864.46],
            [0.020561215, 100.0, 4.0, 26.0, 179.0, 29876.91],
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


    model = SGDLogisticRegression(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        method=args.method,
        epsilon=args.epsilon,
    )
    model.fit(
        X,
        y,
        fit_bias=args.fit_bias,
        save_report=args.save_report,
        report_dir=args.report_dir,
    )

    if args.save_report:
        console.print(
            "[bold green]Training complete and report saved![/bold green]"
        )


if __name__ == "__main__":
    main()

# Example usage:
# python3 sgd_logistic_regression.py --learning_rate 0.5 --max_iter 10 --epsilon 0.01 --save_report --report_dir .
# python3 sgd_logistic_regression.py --learning_rate 1e-8 --max_iter 10 --epsilon 0.01 --method "sgd" --fit_bias --save_report --report_dir .
