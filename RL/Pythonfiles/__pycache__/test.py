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
    def __init__(self, maze_file, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Load the maze
        self.maze = pd.read_csv(maze_file, header=None).to_numpy()
        self.rows, self.cols = self.maze.shape

        # Initialize the Q-table
        self.q_table = np.zeros((self.rows, self.cols, len(DIRECTIONS)))

        # Graph representation of the maze
        self.graph = nx.Graph()
        self._build_graph()
        self.max_steps=500
    def _build_graph(self):
        """Build a graph representation of the maze."""
        for row in range(self.rows):
            for col in range(self.cols):
                if self.maze[row, col] != 0:  # Non-wall cells
                    for action, (dr, dc) in enumerate(DIRECTIONS):
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < self.rows and 0 <= new_col < self.cols and self.maze[new_row, new_col] != 0:
                            self.graph.add_edge((row, col), (new_row, new_col), action=action)

    def choose_action(self, state):
        """Choose the next action using an epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(DIRECTIONS))  # Explore
        row, col = state
        return np.argmax(self.q_table[row, col])  # Exploit

    def step(self, state, action):
        """Take a step in the environment."""
        row, col = state
        dr, dc = DIRECTIONS[action]
        new_row, new_col = row + dr, col + dc

        # Check if the new state is valid
        if 0 <= new_row < self.rows and 0 <= new_col < self.cols and self.maze[new_row, new_col] != 0:
            return (new_row, new_col)
        return state  # If invalid move, remain in the same state

    def get_reward(self, state):
        """Return the reward for the given state."""
        row, col = state
        cell_value = self.maze[row, col]
        if cell_value == 2:  # Hole
            return -1000
        elif cell_value == 3:  # Goal
            return 5000
        elif cell_value == 1:  # Empty cell
            return -10
        return -500  # Walls or invalid moves

    def train(self, n_episodes=1000, max_steps=500):
        for episode in range(n_episodes):
            # Start at a random position
            start_positions = list(zip(*np.where(self.maze == 1)))
            state = random.choice(start_positions)

            for step in range(max_steps):
                action = self.choose_action(state)
                next_state = self.step(state, action)
                reward = self.get_reward(next_state)

                # Update Q-value
                row, col = state
                next_row, next_col = next_state
                best_next_action = np.max(self.q_table[next_row, next_col])

                self.q_table[row, col, action] += self.learning_rate * (
                    reward + self.discount_factor * best_next_action - self.q_table[row, col, action]
                )

                state = next_state

                # Stop if the goal is reached
                if self.maze[next_row, next_col] == 3:
                    break

    def save_q_table(self, filename):
        """Save the Q-table to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.q_table.tolist(), f)

    def visualize_graph(self):
        """Visualize the graph representation of the maze."""
        pos = {node: (node[1], -node[0]) for node in self.graph.nodes}  # Flip y-axis for visualization
        nx.draw(self.graph, pos, with_labels=True, node_size=200, font_size=8)
        plt.show()
    def test_agent(self, start_point):
        """
        Test the agent by making it follow the learned policy to reach the goal.
        """
        print("\nStarting Test Phase")
        current_point = start_point  # Set the initial position
        steps = 0
        path = [current_point]  # Track the agent's path
    
        while steps < self.max_steps:
            row, col = current_point
            action = np.argmax(self.q_table[row, col])  # Choose the best action from Q-table
            next_point = self.step(current_point, action)
    
            # Add the next point to the path
            path.append(next_point)
    
            # Check if the agent has reached the goal
            if self.maze[next_point[0], next_point[1]] == 3:  # Goal cell
                print(f"Goal reached in {steps} steps!")
                break
            
            current_point = next_point
            steps += 1
    
        if steps == self.max_steps:
            print("Test failed: Agent did not reach the goal within the step limit.")
    
        print("Agent's Path:", path)
        return path
    
    def visualize_path(self, path):
        """
        Visualize the maze and the path taken by the agent.
        """
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
        """
        Run the test phase and visualize the agent's path.
        """
        # Choose a starting point (replace with a valid start position)
        start_positions = list(zip(*np.where(self.maze == 1)))
        start_point = random.choice(start_positions)
    
        print(f"Testing from start point: {start_point}")
        path = self.test_agent(start_point)
        self.visualize_path(path)
    
if __name__ == "__main__":
    agent = QLearningAgentWithGraph("RL/matrix.csv")
    agent.train(n_episodes=1000)
    agent.save_q_table("RL\Maze_sim_model\qTable.json")
    agent.visualize_graph()
    agent.run_test()
