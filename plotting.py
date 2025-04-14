import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_model_comparison(csv_path1, csv_path2, model1_name="QYOLOv11", model2_name="YOLOv11"):
    """
    Plot comparison of training metrics between two models.
    
    Args:
        csv_path1 (str): Path to first model's CSV file
        csv_path2 (str): Path to second model's CSV file
        model1_name (str): Name label for first model
        model2_name (str): Name label for second model
    """
    # Read CSV files
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'mAP50-95',
            'mAP50',
            'Validation Box Loss',
            'Validation Class Loss',
            'Validation DFL Loss'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Colors for each model
    color1 = '#1f77b4'  # blue
    color2 = '#ff7f0e'  # orange
    
    # Plot mAP50-95
    fig.add_trace(
        go.Scatter(x=df1['epoch'], y=df1['metrics/mAP50-95(B)'],
                  name=f'{model1_name} mAP50-95',
                  line=dict(color=color1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df2['epoch'], y=df2['metrics/mAP50-95(B)'],
                  name=f'{model2_name} mAP50-95',
                  line=dict(color=color2)),
        row=1, col=1
    )
    
    # Plot mAP50
    fig.add_trace(
        go.Scatter(x=df1['epoch'], y=df1['metrics/mAP50(B)'],
                  name=f'{model1_name} mAP50',
                  line=dict(color=color1)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df2['epoch'], y=df2['metrics/mAP50(B)'],
                  name=f'{model2_name} mAP50',
                  line=dict(color=color2)),
        row=1, col=2
    )
    
    # Plot val/box_loss
    fig.add_trace(
        go.Scatter(x=df1['epoch'], y=df1['val/box_loss'],
                  name=f'{model1_name} Box Loss',
                  line=dict(color=color1)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df2['epoch'], y=df2['val/box_loss'],
                  name=f'{model2_name} Box Loss',
                  line=dict(color=color2)),
        row=2, col=1
    )
    
    # Plot val/cls_loss
    fig.add_trace(
        go.Scatter(x=df1['epoch'], y=df1['val/cls_loss'],
                  name=f'{model1_name} Class Loss',
                  line=dict(color=color1)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df2['epoch'], y=df2['val/cls_loss'],
                  name=f'{model2_name} Class Loss',
                  line=dict(color=color2)),
        row=2, col=2
    )
    
    # Plot val/dfl_loss
    fig.add_trace(
        go.Scatter(x=df1['epoch'], y=df1['val/dfl_loss'],
                  name=f'{model1_name} DFL Loss',
                  line=dict(color=color1)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df2['epoch'], y=df2['val/dfl_loss'],
                  name=f'{model2_name} DFL Loss',
                  line=dict(color=color2)),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        title_text="Model Training Metrics Comparison",
        showlegend=True,
        template="plotly_white"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    fig.update_xaxes(title_text="Epoch", row=3, col=2)
    
    # Update y-axes labels
    metrics = ['mAP50-95', 'mAP50', 'Box Loss', 'Class Loss', 'DFL Loss']
    row_col_pairs = [(1,1), (1,2), (2,1), (2,2), (3,1)]
    
    for (row, col), metric in zip(row_col_pairs, metrics):
        fig.update_yaxes(title_text=metric, row=row, col=col)
    
    return fig

# Example usage:
if __name__ == "__main__":
    # Replace these paths with your actual CSV files
    model1_csv = "runs\\obb\\train5\\results.csv"
    model2_csv = "runs\\obb\\train3\\results.csv"
    
    fig = plot_model_comparison(
        model1_csv,
        model2_csv,
        model1_name="QYOLOv11",
        model2_name="YOLOv11"
    )
    
    # Save the plot
    fig.write_html("model_comparison.html")  # Interactive HTML
    fig.write_image("model_comparison.png")  # Static image