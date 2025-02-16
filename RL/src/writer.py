import json
import numpy as np
from config import Config

class Writer:
    
    def __init__(self):
        self._param=Config()
        self._save_path=self._param._Q_TABLE_PATH
    

    def save_q_table(self,q_table: np.ndarray):
        """Save the Q-table to a JSON file."""
        with open(self._save_path, 'w') as f:
            json.dump(q_table.tolist(), f)