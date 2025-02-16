import pandas as pd
import numpy as np
import json
from config import Config

class Reader:
    
    def __init__(self):
        self.param = Config()  # Initialize Config inside __init__

    def read_maze(self, path=None):
        """Reads the maze from CSV and returns it as a NumPy array."""
        if path is None:
            path = self.param.maze_path  # Use @property directly
        return pd.read_csv(path, header=None).to_numpy()
        
    def read_qtable(self, path=None):
        """Reads the Q-table from a JSON file and returns it as a NumPy array."""
        if path is None:
            path = self.param.q_table_path  # Use @property directly
        with open(path, 'r') as f:
            q_table_list = json.load(f)  
        return np.array(q_table_list)
