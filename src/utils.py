import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config


class Util:
    def __init__(self):
        self._param=Config()
        self._PIC_MAZE_PATH=self._param.maze_pic_path
        self._PIC_MATRIX_RAW=self._param.raw_maze_path

    def pic_to_matrix(self):
        
        image = cv2.imread(self._PIC_MAZE_PATH)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold to binary
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Resize to 60x60 (matrix representation)
        resized = cv2.resize(binary, (60, 60), interpolation=cv2.INTER_AREA)

        # Create matrix: 1 for walls, 0 for paths
        matrix = np.where(resized < 200, 1, 0)  # Adjust threshold as needed

        sns.heatmap(matrix)
        plt.show()
        np.savetxt(self._PIC_MATRIX_RAW, matrix, delimiter=',', fmt='%d')