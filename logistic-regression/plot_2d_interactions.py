import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_2d_interactions(X: pd.DataFrame, features: list, y: pd.Series, bins=10, plot_type='heatmap', title='', figsize=(8, 6), dpi=100):
    """
    Plot 2D interaction between two features based on the observed target variable.

    Parameters:
        X (pd.DataFrame): The input DataFrame containing the features.
        features (list): A list of two feature names to be plotted on the x and y axes.
        y (pd.Series): The target variable to be plotted on the color scale.
        bins (int, optional): The number of bins to use for binning the features. Default is 10.
        plot_type (str, optional): The type of plot to generate. Either 'heatmap' or 'contour'. Default is 'heatmap'.
        title (str, optional): The title of the plot. Default is an empty string.
        figsize (tuple, optional): The size of the figure (width, height). Default is (8, 6).
        dpi (int, optional): Dots per inch for the figure. Default is 80.

    Raises:
        ValueError: If features is not a list containing exactly two feature names.

    Returns:
        None
    """
    # Ensure features is a list of two feature names
    if not isinstance(features, list) or len(features) != 2:
        raise ValueError("features must be a list containing two feature names.")

    # Apply binning to the features
    df = X.copy()
    df[features[0]] = pd.cut(df[features[0]], bins=bins, precision=2)
    df[features[1]] = pd.cut(df[features[1]], bins=bins, precision=2)

    # Calculate the average target values for different combinations of binned features
    average_target_values = df.groupby([features[0], features[1]])[y.name].mean().reset_index()

    # Pivot the data to create a 2D grid for the heatmap
    heatmap_data = average_target_values.pivot(index=features[0], columns=features[1], values=y.name)

    # Convert the bin intervals to numerical values (midpoints)
    heatmap_data.columns = heatmap_data.columns.map(lambda x: x.mid)
    heatmap_data.index = heatmap_data.index.map(lambda x: x.mid)

    # Convert the heatmap_data to numeric type
    heatmap_data = heatmap_data.apply(pd.to_numeric)

    if plot_type == 'heatmap':
        # Create the heatmap
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(heatmap_data.iloc[::-1], cmap='viridis', aspect='auto')
        plt.colorbar(label=y.name)
        plt.title(title)
        plt.xlabel(features[1])
        plt.ylabel(features[0])

        # Set tick positions and labels for x and y axes
        x_ticks = np.arange(len(heatmap_data.columns))
        x_labels = [f"{val:.0f}" for val in heatmap_data.columns]
        plt.xticks(ticks=x_ticks, labels=x_labels, rotation='vertical')

        y_ticks = np.arange(len(heatmap_data.index))
        y_labels = [f"{val:.1f}" for val in heatmap_data.index[::-1]]
        plt.yticks(ticks=y_ticks, labels=y_labels)

        plt.show()
    elif plot_type == 'contour':
        # Create the contour plot
        plt.figure(figsize=figsize, dpi=dpi)
        levels = np.linspace(heatmap_data.min().min(), heatmap_data.max().max(), num=8)
        plt.contourf(heatmap_data.columns, heatmap_data.index[::-1], heatmap_data.values[::-1], levels=levels, cmap='viridis')
        plt.colorbar(label=y.name)
        plt.title(title)
        plt.xlabel(features[1])
        plt.ylabel(features[0])
        plt.show()
    else:
        print("Invalid plot_type. Please choose either 'heatmap' or 'contour'.")