import numpy as np
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import random
import seaborn as sns 


# Define directions
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


class QLearningAgentWithGraph:
    def __init__(self, maze_file,q_table_path):

        # Load the maze
        self.maze = pd.read_csv(maze_file, header=None).to_numpy()
        self.rows, self.cols = self.maze.shape
        self.load_q_table(q_table_path)
        
        self.tothistory = np.zeros((self.rows, self.cols))

        # Graph representation of the maze
        self.graph = nx.Graph()
        self._build_graph()
        self.max_steps=2000
    
    def load_q_table(self, filename):
        with open(filename, 'r') as f:
            q_table_list = json.load(f)  
        self.q_table = np.array(q_table_list)
        
    def _build_graph(self):
        """Build a graph representation of the maze."""
        for row in range(self.rows):
            for col in range(self.cols):
                if self.maze[row, col] != 0:  # Non-wall cells
                    for action, (dr, dc) in enumerate(DIRECTIONS):
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < self.rows and 0 <= new_col < self.cols and self.maze[new_row, new_col] != 0:
                            self.graph.add_edge((row, col), (new_row, new_col), action=action)       
    
    def test_agent(self, start_point):
        
        print("\nStarting Test Phase")
        current_point = start_point  
        steps = 0
        path = [current_point]  
    
        while steps < self.max_steps:
            row, col = current_point
            action = np.argmax(self.q_table[row, col])  
            next_point= self.step_test(current_point, action)
            path.append(next_point)
            if self.maze[next_point[0], next_point[1]] == 3:  # Goal cell
                print(f"Goal reached in {steps} steps!")
                break
            current_point = next_point
            steps += 1
        if steps == self.max_steps:
            print("Test failed: Agent did not reach the goal within the step limit.")
        print("Agent's Path:", path)
        return path
    
    def step_test(self, state, action):
        row, col = state
        dr, dc = DIRECTIONS[action]
        new_row, new_col = row + dr, col + dc
        return (new_row, new_col)
    
    def visualize_path(self, path):

        maze_copy = self.maze.copy()
        # Mark the path on the maze
        for point in path:
            if maze_copy[point[0], point[1]] == 1:  # Empty cell
                maze_copy[point[0], point[1]] = 4  # Mark path cells as 4
        # Plot the maze
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            maze_copy,
            cmap="viridis",
            cbar=False,
            annot=False,
            linewidths=0.5,
            linecolor="black"
        )
    
        # Overlay path points for better visibility
        for point in path:
            plt.scatter(point[1] + 0.5, point[0] + 0.5, c="red", s=100)
    
        plt.title("Agent's Path to Goal")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.show()
    
    def run_test(self):
        # Choose a starting point (replace with a valid start position)
        start_positions = list(zip(*np.where(self.maze == 1)))
        start_point = random.choice(start_positions)
        start_point=[3,1]
        print(f"Testing from start point: {start_point}")
        path = self.test_agent(start_point)
        self.visualize_path(path)    
        

if __name__ == "__main__":
    
    maz_path="RL\solvew by python\matrix_edited.csv"
    q_table_path="RL\solvew by python\qTable.json"
    
    agent = QLearningAgentWithGraph(maz_path,q_table_path)
    agent.run_test()