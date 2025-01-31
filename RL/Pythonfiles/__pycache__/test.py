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
    def __init__(self, maze_file, learning_rate=0.1, discount_factor=0.9, max_epsilon=0.1,min_epsilon=0.05,decay_rate=0.001):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate=decay_rate

        # Load the maze
        self.maze = pd.read_csv(maze_file, header=None).to_numpy()
        self.rows, self.cols = self.maze.shape
        self.load_q_table("RL\Maze_sim_model\qTable.json")
        
        # Initialize the Q-table
        #self.q_table = np.zeros((self.rows, self.cols, len(DIRECTIONS)))
        
        self.tothistory = np.zeros((self.rows, self.cols))

        # Graph representation of the maze
        self.graph = nx.Graph()
        self._build_graph()
        self.max_steps=2000
    def _build_graph(self):
        """Build a graph representation of the maze."""
        for row in range(self.rows):
            for col in range(self.cols):
                if self.maze[row, col] != 0:  # Non-wall cells
                    for action, (dr, dc) in enumerate(DIRECTIONS):
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < self.rows and 0 <= new_col < self.cols and self.maze[new_row, new_col] != 0:
                            self.graph.add_edge((row, col), (new_row, new_col), action=action)

    def choose_action(self, state,episode,History):
        availa_act=[]
        availa_point=[]
        chosek=[]
        for p,item in enumerate(DIRECTIONS):
            nstate,k=self.step(state,p,History)
            if(nstate[0]==state[0])&(nstate[1]==state[1]):
                if (k):
                    chosek.append(p)
                continue
            else:
                availa_act.append(p)
                availa_point.append(self.q_table[state[0]][state[1]][p])
        """Choose the next action using an epsilon-greedy policy."""
        a=self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
        b=random.uniform(0, 1)
        if len(availa_act)>0:
            if b >a:            
                if np.max(availa_point)==0:
                    return random.choice(availa_act)
                return availa_act[np.argmax(availa_point)]  # Exploit
            return random.choice(availa_act)  # Explore
        else:
            return random.choice(chosek)
    def step(self, state, action,History):
        """Take a step in the environment."""
        row, col = state
        dr, dc = DIRECTIONS[action]
        new_row, new_col = row + dr, col + dc
        if len(History)>0:
            if any((int(item[0][0]) == int(new_row)) and (int(item[0][1]) == int(new_col)) for item in History[-10:]):
                return ((row, col),True)
        # Check if the new state is valid
        if 0 <= new_row < self.rows and 0 <= new_col < self.cols and self.maze[new_row, new_col] != 0:
            return ((new_row, new_col),True)
        else:
            return ((row, col),False)  # If invalid move, remain in the same state
    def step_test(self, state, action):
        """Take a step in the environment."""
        row, col = state
        dr, dc = DIRECTIONS[action]
        new_row, new_col = row + dr, col + dc
        return (new_row, new_col)

    def get_reward(self, ostate, state):
        """Return the reward for the given state."""
        row, col = state
        orow,ocol=ostate
        goal_point=list(zip(*np.where(self.maze == 3)))
        cell_value = self.maze[row, col]
        reward_newfound=0
        distance=0
        if cell_value == 2:  # Hole
            reward=-1000
        elif cell_value == 3:  # Goal
            reward= 5000
        elif cell_value == 1:  # Empty cell
            reward= 0
        elif cell_value == 4:
            reward= 0
        else:
            reward=-500
        if self.tothistory[row][col]==0:  
                reward_newfound=10
                ndistance=(abs(goal_point[0][0]-row)+abs(goal_point[0][1]-col))
                odistance=(abs(goal_point[0][0]-orow)+abs(goal_point[0][1]-ocol)) 
                distance= (odistance- ndistance) *10 
        return reward+ distance+reward_newfound # Walls or invalid moves

    def train(self, n_episodes, max_steps):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        heatmap_plot = ax.imshow(self.maze, cmap='hot', interpolation='nearest')

        for episode in range(n_episodes):
            # Start at a random position
            start_positions = list(zip(*np.where(self.maze == 1)))
            state = random.choice(start_positions)
            #state=start_positions[0]
            History=[]
            # Initialize agent marker
            agent_marker, = ax.plot([], [], 'bo')  # 'ro' = red dot
            for step in range(max_steps):
                action = self.choose_action(state,episode,History)
                next_state,_ = self.step(state, action,History)
                reward = self.get_reward(state,next_state)
                row, col = state
                History.append(((row, col),action))
                
                # Update Q-value
                #agent_marker.set_data([col],[row] )
                #plt.draw()
                #plt.pause(0.005)
                next_row, next_col = next_state
                self.tothistory[row][col]=1
                
                best_next_action = np.max(self.q_table[next_row, next_col])

                self.q_table[row, col, action] += self.learning_rate * (
                    reward + self.discount_factor * best_next_action - self.q_table[row, col, action]
                )

                state = next_state

                # Stop if the goal is reached
                if self.maze[next_row, next_col] == 3:
                    seen = set()
                    uHistory = [entry for entry in History if entry not in seen and not seen.add(entry)]
                    mt=(reward)/(len(uHistory))
                    m=1
                    for item in uHistory:
                        self.q_table[item[0][0]][item[0][1]][item[1]] += (mt*m)
                        m+=1
                    print(f"Goal reached in {episode} : {step} steps!")
                    break

                if self.maze[next_row, next_col] == 2:
                    seen = set()
                    uHistory = [entry for entry in History if entry not in seen and not seen.add(entry)]
                    mt=(reward)/(len(uHistory))
                    m=1
                    for item in uHistory:
                        self.q_table[item[0][0]][item[0][1]][item[1]] +=(mt*m)
                        m+=1
                    print(f"hole reached in {episode} : {step} steps!")
                    break
            if  step== max_steps-1:  
                print(f"max loop reached in {episode} : {step} steps!")
        plt.ioff()  # Turn off interactive mode
        plt.show()

    def save_q_table(self, filename):
        """Save the Q-table to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.q_table.tolist(), f)
            
    def load_q_table(self, filename):
        """Load the Q-table from a JSON file."""
        with open(filename, 'r') as f:
            q_table_list = json.load(f)  # Load the list from the file
        self.q_table = np.array(q_table_list)  # Convert the list back to a NumPy array        

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
            next_point= self.step_test(current_point, action)
    
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
        a=1
    
    def run_test(self):
        """
        Run the test phase and visualize the agent's path.
        """
        # Choose a starting point (replace with a valid start position)
        start_positions = list(zip(*np.where(self.maze == 1)))
        start_point = random.choice(start_positions)
        start_point=[3,1]
        print(f"Testing from start point: {start_point}")
        path = self.test_agent(start_point)
        self.visualize_path(path)
    
if __name__ == "__main__":
    agent = QLearningAgentWithGraph("RL/matrix.csv")
    #agent.train(4000,1000)
    #agent.save_q_table("RL\Maze_sim_model\qTable.json")
    #agent.visualize_graph()
    agent.run_test()
