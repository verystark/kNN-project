import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

responses = np.random.randint(0,2,(25,1)).astype(np.float32)

# Label the red = 0 and blue = 1 classes
red = trainData[responses.ravel() == 0]
plt.scatter(red[:, 0], 
            red[:, 1],
            s=80,
            c='r',
            marker='^')

blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:,0],
            blue[:,1],
            s=80,
            c='b',
            marker='s')

newComer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newComer[:,0],
            newComer[:,1],
            s=80,
            c='g',
            marker='o')

knn = cv.ml.KNearest.create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
retval, results, neighborResponses, dist = knn.findNearest(newComer, 3)

print('Red is 0, blue is 1')
print(f'Result: {results}')
print(f'Neighbours: {neighborResponses}')
print(f'Distance: {dist}')

# Save trained model
knn.save('knn_model.yml')

plt.show()
