import subprocess
import os
from rich.console import Console
from rich.table import Table
import ast
import json

# Initialize console for rich output
console = Console()

# List of script files to run
scripts = [
    "ordinal_training_2_layers.py",
    "ordinal_training_3_layers.py",
    "ordinal_training_transformer.py",
    "softmax_training_2_layers.py",
    "softmax_training_3_layers.py",
    "softmax_training_transformer.py",
]

# Initialize a table
table = Table(title="Experiment Results")

# Add columns to the table
table.add_column("Script", justify="right", style="cyan", no_wrap=True)
table.add_column("Best Hyperparameters", style="magenta")
table.add_column("Best Score", justify="right", style="green")
table.add_column("Classification Report", style="yellow")

# Directory containing the scripts
script_dir = "."

# Output file path
output_file = "experiment_results.txt"

# Run each script and collect results
results = []
for script in scripts:
    script_path = os.path.join(script_dir, script)

    # Check if the script file exists
    if not os.path.isfile(script_path):
        console.print(f"[red]Script file {script_path} does not exist![/red]")
        continue

    # Run the script and capture output
    console.print(f"Running script: {script_path}")
    result = subprocess.run(["python3", script_path], capture_output=True, text=True)

    # Check for errors in the execution
    if result.returncode != 0:
        console.print(f"[red]Error running script {script_path}[/red]")
        console.print(result.stderr)
        continue

    # Extract relevant results from the output
    output = result.stdout
    best_params = None
    best_score = None
    classification_report = None

    for line in output.split("\n"):
        if line.startswith("Best hyperparameters:"):
            best_params_str = line.split("Best hyperparameters:")[1].strip()
            best_params = ast.literal_eval(best_params_str)
        elif line.startswith("Best score:"):
            best_score = line.split("Best score:")[1].strip()
        elif "precision" in line:  # Start of the classification report
            classification_report = "\n".join(output.split("\n")[output.split("\n").index(line) :])
            break  # Stop reading lines after the classification report starts

    # Convert best_params to a pretty JSON string
    best_params_str = json.dumps(best_params, indent=4) if best_params else ""

    # Add a row to the table
    table.add_row(script, best_params_str, best_score or "", classification_report or "")

    # Append results for writing to file
    results.append(
        {
            "script": script,
            "best_params": best_params_str,
            "best_score": best_score or "",
            "classification_report": classification_report or "",
        }
    )

# Display the table in the console
console.print(table)

# Write results to a text file
with open(output_file, "w") as f:
    for result in results:
        f.write(f"Script: {result['script']}\n")
        f.write(f"Best Hyperparameters: {result['best_params']}\n")
        f.write(f"Best Score: {result['best_score']}\n")
        f.write("Classification Report:\n")
        f.write(result["classification_report"])
        f.write("\n" + "=" * 80 + "\n")

print(f"Results written to {output_file}")
