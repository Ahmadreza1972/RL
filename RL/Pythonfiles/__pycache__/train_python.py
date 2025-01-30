"""
Train a policy to navigate a specific board configuration using Q-learning
"""

import json
import os
import time
import numpy as np
import random
import pandas as pd
from collections import Counter
from copy import deepcopy
from alpyne.sim import AnyLogicSim
from interactive import print_board
import math

max_steps=500
n_episodes=1000
step=0
class PathfinderTrainer:
    # Do not change the order of these! They're based on the order of the collection in the sim
    DIRECTIONS = ["EAST", "SOUTH", "WEST", "NORTH", "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST"]

    def __init__(self, sim,
                 config_kwargs,
                 lr=0.9,
                 max_steps=max_steps,
                 gamma=0.9,
                 max_epsilon=1.0, min_epsilon=0.05,
                 decay_rate=0.01):
        # model related vars
        self.sim = sim
        self.config_kwargs = config_kwargs

        # rl related vars
        self.lr = lr
        self.max_steps = max_steps
        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        file_path = "RL\Maze_sim_model\qTable.json"
        
        self.hole_reward=-1000
        self.wall_reward=-500
        self.goal_reward=5000
        self.revisit_reward=-100
        self.new_cell_reward=50
        # Open the file and load its contents
        with open(file_path, "r") as file:
            data = json.load(file)
        self.q_table = np.array(data)
        self.q_table=np.zeros((60*60,8))
    def get_epsilon(self, episode: int):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def get_action(self, state: int, episode: int,history,point) -> int:

        matrix = np.array(self.config_kwargs["matrix"])
        availa_act=[]
        availa_point=[]
        for p,item in enumerate(PathfinderTrainer.DIRECTIONS):
            npoint=self.take_action(item,point)
            if((npoint[0]>=60)|(npoint[1]>=60)|(npoint[0]<0)|(npoint[1]<0)):
                continue
            elif (matrix[npoint[0]][npoint[1]]==0):
                    if (history[npoint[0]][npoint[1]]!=0):
                        continue
                    else:
                        if npoint==point:
                            availa_act.append(p)
                            availa_point.append(self.q_table[state][p])
                        else:
                            continue
            else:
                availa_act.append(p)
                availa_point.append(self.q_table[state][p])
                
        if (len(availa_act))==0:
            return 100
        if episode < 0 or random.uniform(0, 1) > self.get_epsilon(episode):
            action = availa_act[np.argmax(availa_point)]
        else:
            action =random.choice(availa_act) #random.randint(0, self.q_table.shape[1] - 1)
        return action

    def get_ini_point(self,n_eps):
        submatrices = []
        p=int(60/math.sqrt(n_eps))
        matrix = np.array(self.config_kwargs["matrix"]) 
        # Loop through the matrix to extract submatrices
        for i in range(0, 60, p):  # Loop through rows
            for j in range(0, 60, p):  # Loop through columns
                
                submatrix = matrix[i:i+p,j:j+p]  # Extract submatrix of size p x q
                submatrices.append((submatrix, i, j))
        ini_pint=[]
        for sub in  submatrices:         
            sub_el, row_offset, col_offset = sub
            positions = [(i, j) for i, row in enumerate(sub_el) for j, value in enumerate(row) if value == 1]
            
            if positions:
                agent_loc_ini=random.choice(positions)
            # Choose a random element from the submatrix
                random_row=agent_loc_ini[0]
                random_col=agent_loc_ini[1]
                # Convert to global coordinates in the original matrix
                global_row = row_offset + random_row
                global_col = col_offset + random_col
                agent_loc_ini=[global_row,global_col]
                if matrix[global_row][global_col]!=1:
                    print("wrong")
                ini_pint.append(agent_loc_ini)
        return ini_pint
    def take_action(self,dir,point):
        Curr_x=point[0]
        Curr_y=point[1]
        if (dir == "EAST"):
            return Curr_x,Curr_y+1
        elif (dir == "SOUTH"):
            return Curr_x-1,Curr_y
        elif (dir == "WEST"):
            return Curr_x,Curr_y-1
        elif (dir == "NORTH"):
            return Curr_x+1,Curr_y
        elif (dir == "NORTHEAST"):
            return Curr_x+1,Curr_y+1
        elif (dir == "NORTHWEST"):
            return Curr_x+1,Curr_y-1
        elif (dir == "SOUTHEAST"):
            return Curr_x-1,Curr_y+1
        elif(dir == "SOUTHWEST"):
            return Curr_x-1,Curr_y-1
        return Curr_x, Curr_y           
    
    def get_reward(self,point):
        matrix = np.array(self.config_kwargs["matrix"])
        if matrix[point[0]][point[1]]==2:
            return self.hole_reward
        if matrix[point[0]][point[1]]==0:
            return self.wall_reward
        if matrix[point[0]][point[1]]==3:
            return self.goal_reward
        if matrix[point[0]][point[1]]==1:
            return self.new_cell_reward
        else:
            return 0
        
    def _execute(self, n_eps, in_train, config_overrides: dict = None, **kwargs):

        
        reward_totals = []

        
        for episode in range(n_eps):
            History=[]
            visit_counter = np.zeros((60, 60))
            do_log = False #log_every > 0 and episode % log_every == 0
            if do_log:
                print(f"\nEPISODE {episode} / {n_eps}")                
            reward_total = 0
            point=[3,1]
            for step in range(self.max_steps):
                stop_eps=False
                
                row, col = point[0],point[1]
                state = row * 60 + col  # 60*60 board
                action = self.get_action(state,
                                         episode if in_train else -1,visit_counter,point)  # use only greedy policy (-1 "episode" in testing)
                if action==100:
                    print(f"ep: {episode}  step{step}  got stock")
                    break
                
                History.append(((row, col),PathfinderTrainer.DIRECTIONS[action]))
                new_status = self.take_action(PathfinderTrainer.DIRECTIONS[action],point)
                
                new_row, new_col = new_status[0],new_status[1]
                
                new_state = new_row * 60 + new_col 
                
                
                reward_revisit=(self.revisit_reward *int(visit_counter[new_row][new_col]))
                visit_counter[new_row][new_col]+=1
                reward = self.get_reward(new_status)
                
                if (reward==self.hole_reward):
                    seen = set()
                    uHistory = [seen.add(entry) for entry in History if entry not in seen ]
                    uHistory=seen
                    mt=(reward*10)/(len(uHistory))
                    m=1
                    for item in uHistory:
                        if in_train:
                            histate=item[0][0]*60+item[0][1]
                            histact=PathfinderTrainer.DIRECTIONS.index(item[1])
                            self.q_table[histate][histact] = self.q_table[histate][histact] +(mt*m)
                            m+=1
                    stop_eps=True
                    
                if (reward==self.goal_reward):
                    seen = set()
                    uHistory = [seen.add(entry) for entry in History if entry not in seen ]
                    uHistory=seen
                    mt=reward/(len(History))
                    m=1
                    for item in History:
                        if in_train:
                            histate=item[0][0]*60+item[0][1]
                            histact=PathfinderTrainer.DIRECTIONS.index(item[1])
                            self.q_table[histate][histact] = self.q_table[histate][histact] + (mt*m)
                            m+=1
                    stop_eps=True
                
                
                reward_wall_dirict=0    
                if ((row==new_row) & (new_col==col)):
                    reward_wall_dirict= self.wall_reward
                    
                reward+= (reward_wall_dirict+reward_revisit)  
                reward_total += reward

                if in_train:
                    self.q_table[state][action] = self.q_table[state][action] + self.lr * (
                            reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[state][action])

                point = new_status
                if (stop_eps):
                    print(f"ep: {episode}  step{step}")
                    break
                if step==max_steps-1:
                    print(f"ep: {episode}  step{step}")
            reward_totals.append(reward_total)
            if do_log:
                print(f"Score counts: {dict(Counter(reward_totals))} | Epsilon: {self.get_epsilon(episode):.3f}\n\n")
                
        return reward_totals

    def train(self, n_eps, **kwargs):
        return self._execute(n_eps, True, **kwargs)

    def test(self, n_eps, config_overrides: dict = None, **kwargs):
        return self._execute(n_eps, False, config_overrides=config_overrides, **kwargs)


if __name__ == "__main__":
    assert os.path.exists(r"RL/Maze_LR/model.jar"), r"Missing file 'ModelExported/model.jar'. To fix, create the folder if it does not exist and export/unzip in-place."

    sim = AnyLogicSim(r"RL/Maze_LR/model.jar", engine_overrides=dict(seed=147))
    print(sim.schema)
    print("---------")

    start = time.time()

    random.seed(0)
    num_walls=0
    num_holes=0
    num_goals=0
    df=pd.read_csv("RL\matrix.csv",header=None)
    matrixs = df.to_numpy()
    for i in range(0,matrixs.shape[0]):
        for j in range(0,matrixs.shape[1]):
            if matrixs[i][j]==1:
                num_walls+=1
            elif matrixs[i][j]==2:
                num_holes+=1
            elif matrixs[i][j]==3:
                num_goals+=1
    new_matrixs= [row.tolist() for row in matrixs]
    slipchance_tr=0.2
    config = dict(slipchance=slipchance_tr,num_goal=num_goals,num_wall=num_walls,num_hole=num_holes,matrix=new_matrixs,max_step=max_steps,curr_step=step,agent_loc=[])
    trainer = PathfinderTrainer(sim, config,
                                lr=0.7,
                                gamma=0.6,
                                decay_rate=0.005
                                )

    rewards_per_eps = trainer.train(n_episodes, log_every=50, verbose_log=False, print_initial_board=True)

    with open(r"RL/Maze_sim_model/qTable.json", "w") as f:  # point to/move this file in the model to have it be loaded
        json.dump(trainer.q_table.tolist(), f)

    print("Count of reward occurrence:", Counter(rewards_per_eps))
    print(f"Seconds to train: {time.time() - start}")
    #print("Test reward results (no slipping):", Counter(trainer.test(10, config_overrides=dict(slipchance=0))))
    #print("Test reward results (same config):", Counter(trainer.test(10)))
    #print("Test reward results (2x slipping):",Counter(trainer.test(10, config_overrides=dict(slipChance=config['slipchance'] * 2))))
