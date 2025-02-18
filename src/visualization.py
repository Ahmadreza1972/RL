import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from reader import Reader
from config import Config

    
class Visualization:
    def __init__(self):
        
        self._reader=Reader()
        self._param=Config()
        
    def visualize_graph(self):
        """Visualize the maze as a grid."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self._reader.read_maze(), cmap="gray_r", origin="upper")  # Use 'gray_r' for better contrast
        plt.colorbar(label="Cell Type")
        plt.title("Maze Visualization")
        plt.show()
        
    def show_qtable(self):    
        data=self._reader.read_qtable()
        DIRECTIONS=self._param.directions
        NUM_COL=data.shape[1]
        NUM_ROW=data.shape[0]
        ndata = np.zeros((NUM_ROW, NUM_COL))
        directions = np.zeros((NUM_ROW, NUM_COL, 2))  # To store the direction vectors
        # Process the data to calculate mean and direction of max value
        for row in range(NUM_ROW):
            for col in range(NUM_COL):
                i=((row*NUM_ROW)+col)//NUM_COL
                j=((row*NUM_ROW)+col)%NUM_COL
                nr=((row*NUM_ROW)+col)
                ndata[i][j] = np.average(data[i][j])  # Compute the mean
                max_index = np.argmax(data[i][j])  # Index of the maximum value
                directions[j, i] = DIRECTIONS[max_index]
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(ndata, cmap="viridis", annot=False, cbar=True,vmax=5000,vmin=-5000)
        plt.title("Q-Value Heatmap")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.show()