import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

responses = np.random.randint(0,2,(25,1)).astype(np.float32)

red = trainData[responses.ravel() == 0]
plt.scatter(red[:, 0], 
            red[:, 1],
            80,
            'r',
            '^')

blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:,0],
            blue[:,1],
            80,
            'b',
            's')

plt.show()
