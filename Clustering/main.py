import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib import animation
import seaborn as sns

# Load the dataset
# Assuming the dataset file is named 'students003.txt'
data = pd.read_csv('data\TrajectoryData_students003\students003.txt', sep='\t', header=None, names=['Time', 'ID', 'X', 'Y'])

# Function to cluster and visualize each frame
def cluster_and_visualize(data, epsilon=1.5, min_samples=2):
    # Extract unique timestamps
    timestamps = sorted(data['Time'].unique())

    # Setting up the plot
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame_idx):
        ax.clear()
        timestamp = timestamps[frame_idx]
        frame_data = data[data['Time'] == timestamp]
        X = frame_data[['X', 'Y']].values

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
        labels = clustering.labels_

        # Visualize
        unique_labels = set(labels)
        colors = sns.color_palette('hsv', len(unique_labels))
        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'k'  # Black used for noise
            mask = (labels == label)
            ax.scatter(X[mask, 0], X[mask, 1], c=[color], label=f'Group {label}', s=50)

        ax.set_title(f'Frame {frame_idx+1}, Time {timestamp}')
        ax.legend(loc='upper right')

    anim = animation.FuncAnimation(fig, update, frames=len(timestamps), interval=500, repeat=False)
    plt.close()
    return anim

# Create the animation
group_animation = cluster_and_visualize(data)

# Save the animation as a video file (optional)
group_animation.save('data\group_discovery_animation.mp4', writer='ffmpeg', fps=2)

# For an inline visualization if you're using a notebook environment
from IPython.display import HTML
HTML(group_animation.to_jshtml())
