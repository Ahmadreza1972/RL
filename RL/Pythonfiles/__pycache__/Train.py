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

max_steps=500
n_episodes=100
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
        self.q_table = np.zeros((60*60, 8))

    def get_epsilon(self, episode: int):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def get_action(self, state: int, episode: int) -> int:
        # episode < 0 == testing/evaluation == always use greedy
        if episode < 0 or random.uniform(0, 1) > self.get_epsilon(episode):
            action = np.argmax(self.q_table[state])
        else:
            action = random.randint(0, self.q_table.shape[1] - 1)
        return action

    def _execute(self, n_eps, in_train, config_overrides: dict = None, **kwargs):
        print_initial_board = kwargs.get('print_initial_board', False)
        log_every = kwargs.get('log_every', 1)
        verbose_log = kwargs.get('verbose_log', False)
        visit_counter = np.zeros((60, 60))
        reward_totals = []
        for episode in range(n_eps):
            
            do_log = False #log_every > 0 and episode % log_every == 0
            if do_log:
                print(f"\nEPISODE {episode} / {n_eps}")

            #reset the environment, using default engine engine_settings
            this_config = deepcopy(self.config_kwargs)

            positions = [(i, j) for i, row in enumerate(this_config["matrix"]) for j, value in enumerate(row) if value == 1]
            agent_loc_ini=random.choice(positions)
            this_config.update({"agent_loc":agent_loc_ini})
            if config_overrides:
                this_config.update(config_overrides)
            status = self.sim.reset(**this_config)

            #if episode == 0 and print_initial_board:
                #print_board(status)

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
                action = self.get_action(state,
                                         episode if in_train else -1)  # use only greedy policy (-1 "episode" in testing)
                #if status.state != 'PAUSED':  # Example check for a specific state
                #    print(f"Current engine state: {status.state}")

                new_status = self.sim.take_action(dir=PathfinderTrainer.DIRECTIONS[action])
                new_row, new_col = new_status.observation['pos']
                new_state = new_row * 60 + new_col  # 8x8 board
                visit_counter[new_row][new_col]+=1
                reward = status.observation['cells'][new_row][new_col]
                if ((reward==1000) | (reward==-100)):
                    stop_eps=True
                reward_revisit=0#(-2*visit_counter[new_row][new_col])
                reward_wall_dirict=0
                if ((row==new_row) & (new_col==col)):
                    reward_wall_dirict= -10
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
                if step==n_eps-1:
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
