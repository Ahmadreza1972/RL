import seaborn as sns
import json
import numpy as np
import matplotlib.pyplot as plt

# Specify the file path
file_path = "RL/Maze_sim_model/qTable.json"
DIRECTIONS = [
    (0, 1),  # Right
    (0, -1), # Left
    (1, 0),  # Down
    (-1, 0), # Up
    (1, 1),  # Down-Right
    (-1, -1),# Up-Left
    (1, -1), # Down-Left
    (-1, 1)  # Up-Right
]
# Open the file and load its contents
with open(file_path, "r") as file:
    data = json.load(file)

ndata = np.zeros((60, 60))
directions = np.zeros((60, 60, 2))  # To store the direction vectors

# Process the data to calculate mean and direction of max value
for i in range(60):
    for j in range(60):
        ndata[j][i] = np.average(data[i][j])  # Compute the mean
        max_index = np.argmax(data[i][j])  # Index of the maximum value
        if max_index == 0: 
            directions[j, i] = DIRECTIONS[0]
        elif max_index == 1: 
            directions[j, i] = DIRECTIONS[1]
        elif max_index == 2:  
            directions[j, i] = DIRECTIONS[2]
        elif max_index == 3: 
            directions[j, i] = DIRECTIONS[3]
        elif max_index == 4:  
            directions[j, i] = DIRECTIONS[4]
        elif max_index == 5:  
            directions[j, i] = DIRECTIONS[5]
        elif max_index == 6: 
            directions[j, i] = DIRECTIONS[6]
        elif max_index == 7:  
            directions[j, i] = DIRECTIONS[7]
# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(ndata, cmap="viridis", annot=False, cbar=True)#,vmax=500,vmin=-500
plt.title("Q-Value Heatmap")
plt.xlabel("Column Index")
plt.ylabel("Row Index")

# Overlay arrows to indicate directions
for i in range(60):
    for j in range(60):
        dy, dx = directions[i, j]
        if dx != 0 or dy != 0:  # Only draw arrows for valid directions
            plt.arrow(
                j + 0.5,  # Arrow start x
                i + 0.5,  # Arrow start y
                dx * 0.3,  # Arrow delta x
                -dy * 0.3,  # Arrow delta y (invert y for plotting)
                color="red",
                head_width=0.2,
                head_length=0.2,
                linewidth=0.5
            )

plt.show()
