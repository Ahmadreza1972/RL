import os

class Config:
    """Configuration settings for the project."""
    
    def __init__(self):
        # Get the base directory of the script
        self._BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Data directories (private)
        self._DATA_DIR = os.path.join(self._BASE_DIR, "../data")
        self._RAW_DIR=os.path.join(self._DATA_DIR, "raw")
        self._PROCESSED_DIR = os.path.join(self._DATA_DIR, "processed")
        self._OUTPUT_DIR = os.path.join(self._DATA_DIR, "output")
        self._PIC_DIR = os.path.join(self._DATA_DIR, "pic")

        # File paths (private)
        self._RAW_MAZE_PATH=os.path.join(self._RAW_DIR, "matrix_orginal.csv")
        self._MAZE_PATH = os.path.join(self._PROCESSED_DIR, "edited_maze.csv")
        self._Q_TABLE_PATH = os.path.join(self._OUTPUT_DIR, "qTable.json")
        self._MAZE_PIC_PATH=os.path.join(self._PIC_DIR, "maze.png")
        

        # Define movement directions (public)
        self._DIRECTIONS = [
            (0, 1),   # Right
            (0, -1),  # Left
            (1, 0),   # Down
            (-1, 0),  # Up
            (1, 1),   # Down-Right
            (-1, -1), # Up-Left
            (1, -1),  # Down-Left
            (-1, 1)   # Up-Right
        ]
        
        # Training Hyperparameters
        self._EPOCHS = 4000
        self._MAX_STEP = 1000
        self._LEARNING_RATE=0.1
        self._DISCOUNT_FACTOR=0.9
        self._MAX_EPSILON=0.1
        self._MIN_EPSILON=0.05
        self._DECAY_RATE=0.001
        self._AGENT_MEMORY=10
        self._GOAL_POINT=5000
        self._HOLE_POINT=-1000
        self._PATH_POINT=0
        self._NEW_FOUND_CELL=10
        self._DISTANCE_POIN=10
        
        
        #maze structure in matrix
        self._WALL_REP=0
        self._PATH_REP=1
        self._HOLE_REP=2
        self._GOAL_REP=3
        
    # Public getters for file paths
    @property
    def raw_maze_path(self):
        return self._RAW_MAZE_PATH
    
    @property
    def maze_path(self):
        return self._MAZE_PATH
    
    @property
    def q_table_path(self):
        return self._Q_TABLE_PATH
    
    @property
    def maze_pic_path(self):
        return self._MAZE_PIC_PATH
    
    @property
    def epochs(self):
        return self._EPOCHS

    @property
    def max_step(self):
        return self._MAX_STEP
    
    @property
    def directions(self):
        return self._DIRECTIONS
    
    @property
    def learning_rate(self):
        return self._LEARNING_RATE
    
    @property
    def discount_factor(self):
        return self._DISCOUNT_FACTOR
    
    @property
    def max_epsilon(self):
        return self._MAX_EPSILON
    
    @property
    def min_epsilon(self):
        return self._MIN_EPSILON
    
    @property
    def decay_rate(self):
        return self._DECAY_RATE
    
    @property
    def wall_rep(self):
        return self._WALL_REP
    
    @property
    def path_rep(self):
        return self._PATH_REP
    
    @property
    def hole_rep(self):
        return self._HOLE_REP
    
    @property
    def goal_rep(self):
        return self._GOAL_REP
    
    @property
    def agent_memory(self):
        return self._AGENT_MEMORY
    
    @property
    def goal_point(self):
        return self._GOAL_POINT
    
    @property
    def hole_point(self):
        return self._HOLE_POINT
    
    @property
    def path_point(self):
        return self._PATH_POINT
    
    @property
    def new_found_cell(self):
        return self._NEW_FOUND_CELL
    
    def distance_point(self):
        return self._DISTANCE_POIN
