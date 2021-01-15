import csv
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np

distanceData = genfromtxt('conversion.csv', delimiter = ',')
# print(distanceData)
inputX = distanceData[:,0]
inputy = distanceData[:,1]


# plt.scatter(X,y, c = "red",alpha=.5, marker = 'o')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

def hypothesis(X, weights):
    return (weights[1] * np.array(X) + weights[0])


def getCost(weights, X, y):
    return (.5/len(X)) * np.sum(np.square(hypothesis(X,weights) - np.array(y)))

def getGradient(weights, X, y):
    grad_arr = [0, 0] 
    grad_arr[0] = (1/len(X)) * np.sum(hypothesis(X,weights) - np.array(y))
    grad_arr[1] = (1/len(X)) * np.sum((hypothesis(X,weights) - np.array(y)) * np.array(X))
    return grad_arr

def minimize(weightsNew, weightsPrev, learningRate):
    print(weightsPrev)
    print(getCost(weightsPrev,inputX,inputy))
    iterations=0
    while True:
        weightsPrev = weightsNew
        w0 = weightsPrev[0] - learningRate * getGradient(weightsPrev,inputX,inputy)[0]
        w1 = weightsPrev[1] - learningRate * getGradient(weightsPrev,inputX,inputy)[1]
        weightsNew = [w0, w1]
        print(weightsNew)
        print(getCost(weightsNew,inputX,inputy))
        if (weightsNew[0]-weightsPrev[0])**2 + (weightsNew[1]-weightsPrev[1])**2 <= pow(10,-6):
            return weightsNew
        if iterations>500: 
            return weightsNew
        iterations+= 1

weights = [0, -1]
weightsMinimized = minimize(weights, weights, .1)
print(minimizedWeights)

    
def plotRegression(x):
    return minimizedWeights[0]+minimizedWeights[1]*x

regressionLine = np.array(range(-5,60)) 
plt.scatter(inputX,inputy, c = "red", alpha=.9, marker = 'o')
plt.axis([0, 100, 0, 100])
plt.plot(regressionLine, plotRegression(regressionLine))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
