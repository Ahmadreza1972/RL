import seaborn as sns
import json
import numpy as np
import matplotlib.pyplot as plt

# Specify the file path
file_path = "RL/solvew by python/qTable.json"
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
for row in range(60):
    for col in range(60):
        i=((row*60)+col)//60
        j=((row*60)+col)%60
        nr=((row*60)+col)
        
        ndata[i][j] = np.average(data[i][j])  # Compute the mean
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
sns.heatmap(ndata, cmap="viridis", annot=False, cbar=True,vmax=5000,vmin=-5000)
plt.title("Q-Value Heatmap")
plt.xlabel("Column Index")
plt.ylabel("Row Index")

plt.show()
