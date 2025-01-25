import seaborn as sns 
import json
import numpy as np
import matplotlib.pylab as plt
# Specify the file path
file_path = "RL\Maze_sim_model\qTable.json"
# Open the file and load its contents
with open(file_path, "r") as file:
    data = json.load(file)

ndata=np.zeros((60,60))
# Print the loaded data
for i in range(0,60):
    for j in range(0,60):
        ndata[j][i]=max(data[i*60+j])

plt.figure(figsize=(10, 8))
sns.heatmap(ndata, cmap="viridis", annot=False, cbar=True)
plt.title("Q-Value Heatmap")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.show()