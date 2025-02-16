import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
import random
import seaborn as sns 
from config import Config
from reader import Reader


class QLearningAgentWithGraph:
    """A reinforcement learning agent for solving maze-like environments using Q-learning."""
    
    def __init__(self):
        """Initialize agent parameters, Q-table, and graph representation."""
        self._param = Config()
        self._read_files = Reader()
        
        # Set hyperparameters
        self._learning_rate:float  = self._param.learning_rate
        self._discount_factor:float  = self._param.discount_factor
        self._max_epsilon:float  = self._param.max_epsilon
        self._min_epsilon:float  = self._param.min_epsilon
        self._decay_rate:float  = self._param.decay_rate
        self._epocs:int =self._param.epochs
        self._max_step:int =self._param.max_step
        self._agent_memory:int =self._param.agent_memory
        
        # Set maze-related parameters
        self._MAZE: np.ndarray = self._read_files.read_maze()
        self._DIRECTIONS: list = self._param.directions
        self._WALL_REP:int =self._param.wall_rep
        self._HOLE_REP:int =self._param.hole_rep
        self._GOAL_REP:int =self._param.goal_rep
        self._PATH_REP:int =self._param.path_rep
        self._GOAL_POINTS:int =self._param.goal_point
        self._HOLE_POINTS:int =self._param.hole_point
        self._PATH_POINTS:int =self._param.path_point
        self._NEW_FOUND_POINTS:int =self._param.new_found_cell
        self._DISTANCE_POINT:int =self._param._DISTANCE_POIN
        
        # Corrected shape access
        self._NUM_ROWS, self._NUM_COLS = self._MAZE.shape
        self._NUM_DIR = len(self._DIRECTIONS)
        
        # Initialize matrices
        self._tothistory = np.zeros((self._NUM_ROWS, self._NUM_COLS))
        self._q_table = np.zeros((self._NUM_ROWS, self._NUM_COLS, self._NUM_DIR))
                  

    def choose_action(self, state: tuple, episode: int, history: list)-> int:
        """Choose an action using an epsilon-greedy policy."""
        
        availa_act,availa_point,chose_hist=[],[],[]
        
        # find all possible movements
        for p,_ in enumerate(self._DIRECTIONS):
            nstate,k=self.step(state,p,history)
            if(nstate[0]==state[0])&(nstate[1]==state[1]):
                if (k):
                    chose_hist.append(p)
                continue
            else:
                availa_act.append(p)
                availa_point.append(self._q_table[state[0]][state[1]][p])
                
        decay_func=self._min_epsilon + (self._max_epsilon - self._min_epsilon) * np.exp(-self._decay_rate * episode)
        uni_rrand=random.uniform(0, 1)
        
        # choose an action based on the situation 
        if len(availa_act)>0:
            if uni_rrand >decay_func:            
                if np.max(availa_point)==0:
                    return random.choice(availa_act) # Explore  (as long as, there is unvisited cells, chance for random choose)
                return availa_act[np.argmax(availa_point)]  # Exploit
            return random.choice(availa_act)  # Explore
        else:
            return random.choice(chose_hist) # randomly chose one of the previously visited cell

        
    def step(self, state: tuple, action: int, history: list)-> tuple:
        """Take a step in the environment."""
        row, col = state
        dr, dc = self._DIRECTIONS[action]
        new_row, new_col = row + dr, col + dc
        
        ## check if in last n state, had visited this state
        if len(history)>0:
            if any((int(item[0][0]) == int(new_row)) and (int(item[0][1]) == int(new_col)) for item in history[-self._agent_memory:]):
                return ((row, col),True)
            
        # Check if the new state is valid
        if 0 <= new_row < self._NUM_ROWS and 0 <= new_col < self._NUM_COLS and self._MAZE[new_row, new_col] != 0:
            return ((new_row, new_col),True)
        else:
            return ((row, col),False)  # If invalid move, remain in the same state

    def get_reward(self, old_state: tuple, state: tuple) -> float:
        """Return the reward for the given state."""
        
        reward_newfound=0
        distance=0
        
        row, col = state
        orow,ocol=old_state
        
        goal_point=list(zip(*np.where(self._MAZE == self._GOAL_REP)))
        
        cell_value = self._MAZE[row, col]
        
        reward = {
            self._HOLE_REP: self._HOLE_POINTS,
            self._GOAL_REP: self._GOAL_POINTS,
            self._PATH_REP: self._PATH_POINTS
        }.get(cell_value, 0)

        if self._tothistory[row][col]==0:  
                reward_newfound=self._NEW_FOUND_POINTS
                ndistance=(abs(goal_point[0][0]-row)+abs(goal_point[0][1]-col))
                odistance=(abs(goal_point[0][0]-orow)+abs(goal_point[0][1]-ocol)) 
                distance= (odistance- ndistance) *self._DISTANCE_POINT 
        return reward+ distance+reward_newfound # Walls or invalid moves

    def train(self, show_result: bool = False):
        
        if show_result:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            heatmap_plot = ax.imshow(self.maze, cmap='hot', interpolation='nearest')
            plt.show()
            
        start_positions = list(zip(*np.where(self._MAZE == self._PATH_REP)))    

        for episode in range(self._epocs):
            
            # variables
            history=[]
            
            # Start at a random position
            state = random.choice(start_positions)
            
            # Initialize agent marker
            if show_result:
                agent_marker, = ax.plot([], [], 'bo')  # 'ro' = red dot
                
            for step in range(self._max_step):
                action = self.choose_action(state,episode,history)
                next_state,_ = self.step(state, action,history)
                reward = self.get_reward(state,next_state)
                
                row, col = state
                history.append(((row, col),action))
                
                if show_result:
                    agent_marker.set_data([col],[row] )
                    plt.draw()
                    plt.pause(0.005)
                    
                next_row, next_col = next_state
                self._tothistory[row][col]=1

                best_next_action = np.max(self._q_table[next_row, next_col])
                self._q_table[row, col, action] += self._learning_rate * (
                    reward + self._discount_factor * best_next_action - self._q_table[row, col, action]
                )

                state = next_state
                
                if ((self._MAZE[next_row, next_col] == self._GOAL_REP) |(self._MAZE[next_row, next_col] == self._HOLE_REP)):
                    seen = set()
                    uHistory = [entry for entry in history if entry not in seen and not seen.add(entry)]
                    mt=(reward)/(len(uHistory))
                    m=1
                    for item in uHistory:
                        self._q_table[item[0][0]][item[0][1]][item[1]] +=(mt*m)
                        m+=1

                # Stop if the goal is reached
                if self._MAZE[next_row, next_col] == self._GOAL_REP:
                    print(f"Goal reached in {episode} : {step} steps!")
                    break
                if self._MAZE[next_row, next_col] == self._HOLE_REP:
                    print(f"hole reached in {episode} : {step} steps!")
                    break
                
            if  step== self._max_step-1:  
                print(f"max loop reached in {episode} : {step} steps!")
                
        plt.ioff()  # Turn off interactive mode
        plt.show()

            
    

    
