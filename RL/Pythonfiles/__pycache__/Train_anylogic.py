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
n_episodes=400
step=0
class PathfinderTrainer:
    # Do not change the order of these! They're based on the order of the collection in the sim
    DIRECTIONS = ["EAST", "SOUTH", "WEST", "NORTH", "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST"]

    def __init__(self, sim,
                 config_kwargs,
                 lr=0.7,
                 max_steps=max_steps,
                 gamma=0.6,
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
        # Open the file and load its contents
        with open(file_path, "r") as file:
            data = json.load(file)
        self.q_table = np.array(data)

    def get_epsilon(self, episode: int):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def get_action(self, state: int, episode: int) -> int:

        if episode < 0 or random.uniform(0, 1) > self.get_epsilon(episode):
            action = np.argmax(self.q_table[state])
        else:
            availa_act=[i for i,item in enumerate(self.q_table[state]) if item<-500 ]
            action =random.choice(availa_act) 
        return action

    def _execute(self, n_eps, in_train, config_overrides: dict = None, **kwargs):
        verbose_log = kwargs.get('verbose_log', False)
        visit_counter = np.zeros((60, 60))
        reward_totals = []
        secontlayer=False
        submatrices = []
        p=int(60/math.sqrt(n_eps)) 
        # Loop through the matrix to extract submatrices
        for i in range(0, 60, p):  # Loop through rows
            for j in range(0, 60, p):  # Loop through columns
                matrix = np.array(self.config_kwargs["matrix"])
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

        for episode in range(n_eps):
            History=[]
            do_log = False #log_every > 0 and episode % log_every == 0
            if do_log:
                print(f"\nEPISODE {episode} / {n_eps}")
            this_config = deepcopy(self.config_kwargs)
            
            if secontlayer==False:
                if episode>len(ini_pint)-1:
                    point=random.choice(ini_pint)
                else:
                    point=ini_pint[episode]
            else:
                point=[3,1]
                
            this_config.update({"agent_loc":point})
            if config_overrides:
                this_config.update(config_overrides)
            status = self.sim.reset(**this_config)

            reward_total = 0

            for step in range(self.max_steps):
                stop_eps=False
                if do_log:
                    if verbose_log:
                        print_board(status)
                    else:
                        print(f"\tSTEP {step:2d} | {str(status.stop):5s} | {str(status.observation['pos']):7s}")

                if status.stop:
                    r, c = status.observation['pos']
                    final_reward = status.observation['cells'][r][c]
                    break
                row, col = status.observation['pos']
                state = row * 60 + col  # 60*60 board
                action = self.get_action(state,episode if in_train else -1)
                History.append(((row, col),PathfinderTrainer.DIRECTIONS[action]))
                new_status = self.sim.take_action(dir=PathfinderTrainer.DIRECTIONS[action])
                new_row, new_col = new_status.observation['pos']
                new_state = new_row * 60 + new_col  # 8x8 board
                visit_counter[new_row][new_col]+=1
                reward = status.observation['cells'][new_row][new_col]
                seen = set()
                History = [entry for entry in History if entry not in seen and not seen.add(entry)]        
                        
                if (reward==-100):

                    mt=(reward*10)/(len(History))
                    m=1
                    for item in History:
                        if in_train:
                            histate=item[0][0]*60+item[0][1]
                            histact=PathfinderTrainer.DIRECTIONS.index(item[1])
                            self.q_table[histate][histact] = self.q_table[histate][histact] + self.lr * (
                               (mt*m)+ self.gamma * np.max(self.q_table[histate]) - self.q_table[histate][histact])
                            m+=1
                    stop_eps=True
                if (reward==1000):
                    for item in History:
                        if in_train:
                            mt=reward/(len(History))
                            m=1
                            histate=item[0][0]*60+item[0][1]
                            histact=PathfinderTrainer.DIRECTIONS.index(item[1])
                            self.q_table[histate][histact] = self.q_table[histate][histact] + self.lr * (
                        (mt*m) + self.gamma * np.max(self.q_table[histate]) - self.q_table[histate][histact])
                    stop_eps=True                      
                reward_revisit=(-10 *visit_counter[new_row][new_col])
                reward_wall_dirict=0
                if ((row==new_row) & (new_col==col)):
                    reward_wall_dirict= -1000
                reward+=  (reward_wall_dirict+reward_revisit)  
                reward_total += reward

                if do_log:
                    print(f"\t\t-> {PathfinderTrainer.DIRECTIONS[action]} ({action}) => + {reward} = {reward_total}")

                if in_train:
                    self.q_table[state][action] = self.q_table[state][action] + self.lr * (
                            reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[state][action])

                status = new_status
                if (stop_eps):
                    print(f"ep: {episode}  step{step}")
                    break
                if step==max_steps-1:
                    print(f"ep: {episode}  step{step}")
                    seen = set()
                    History = [entry for entry in History if entry not in seen and not seen.add(entry)] 
                    mt=-1000/(len(History))
                    m=1 
                    for item in History:
                        if in_train:
                            histate=item[0][0]*60+item[0][1]
                            histact=PathfinderTrainer.DIRECTIONS.index(item[1])
                            self.q_table[histate][histact] = self.q_table[histate][histact] + self.lr * (
                        (mt*m) + self.gamma * np.max(self.q_table[histate]) - self.q_table[histate][histact])
                            m+=1                    

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

    with open(r"RL/Maze_sim_model/qTable.json", "w") as f: 
        json.dump(trainer.q_table.tolist(), f)

    print("Count of reward occurrence:", Counter(rewards_per_eps))
    print(f"Seconds to train: {time.time() - start}")
