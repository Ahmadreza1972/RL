import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json as jsn
import pandas as pd
# Load the image
image_path = "RL\maze.png"  # Update with your file path if necessary
image = cv2.imread(image_path)

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
np.savetxt('matrix.csv', matrix, delimiter=',', fmt='%d')