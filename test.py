import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics_from_csv(csv_filepath):
    """
    Plot specific metrics from a CSV file with separate subplots.
    
    Args:
        csv_filepath: Path to the CSV file containing the data
    """
    # Read the CSV file
    df = pd.read_csv(csv_filepath)
    
    # Columns to plot
    metrics = ['metrics/mAP50(B)', 'val/box_loss', 'val/cls_loss', 'lr/pg0']
    
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Check if we have an epoch or iteration column
    x_column = 'epoch' if 'epoch' in df.columns else 'iteration'
    if x_column not in df.columns:
        # If neither exists, create an index
        x_values = np.arange(len(df))
        x_label = 'Index'
    else:
        x_values = df[x_column]
        x_label = x_column.capitalize()
    
    # Plot each metric in its own subplot
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            axes[i].plot(x_values, df[metric], 'o-', linewidth=2, markersize=4)
            axes[i].set_title(metric)
            axes[i].set_xlabel(x_label)
            axes[i].set_ylabel('Value')
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
            # Add a bit of styling
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        else:
            axes[i].text(0.5, 0.5, f"Column '{metric}' not found in CSV", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[i].transAxes, fontsize=12)
    
    # Add a main title
    plt.suptitle('Training Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title
    
    # Show the plot
    plt.show()
    
    # You can also save the figure
    plt.savefig('training_metrics.png')

# Example usage
if __name__ == "__main__":
    csv_file = "runs\\detect\\train73\\results.csv"
    plot_metrics_from_csv(csv_file)