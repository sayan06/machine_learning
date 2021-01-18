import csv
import math
import numpy as numpy
from numpy import genfromtxt
import matplotlib.pyplot as plt

distanceData = genfromtxt('conversion.csv', delimiter=',')

inputX = distanceData[:, 0]
inputy = distanceData[:, 1]


def hypothesis(X, weights):
    return (weights[1] * numpy.array(X) + weights[0])


def getCost(weights, X, y):
    return (.5/len(X)) * numpy.sum(numpy.square(hypothesis(X, weights) - numpy.array(y)))


def getGradient(weights, X, y):
    gradientArray = [0, 0]
    gradientArray[0] = (1/len(X)) * numpy.sum(hypothesis(X, weights) - numpy.array(y))
    gradientArray[1] = (1/len(X)) * \
        numpy.sum((hypothesis(X, weights) - numpy.array(y)) * numpy.array(X))
    return gradientArray


def minimize(weightsNew, weightsPrev, learningRate):

    print(weightsPrev)
    print(getCost(weightsPrev, inputX, inputy))
    
    iterations = 0
    keepIterating = True
    while keepIterating:
        weightsPrev = weightsNew
        w0 = weightsPrev[0] - learningRate * \
            getGradient(weightsPrev, inputX, inputy)[0]
        w1 = weightsPrev[1] - learningRate * \
            getGradient(weightsPrev, inputX, inputy)[1]
        weightsNew = [w0, w1]
    
        print(weightsNew)
        print(getCost(weightsNew, inputX, inputy))
        
        if ( getCost(weightsPrev, inputX, inputy) - getCost(weightsNew, inputX, inputy)) <= pow(10, -2):
            print('reached a  weight diff')
            # keepIterating = False
            return weightsNew

        # if (weightsNew[0] - weightsPrev[0])** 2 + (weightsNew[1] - weightsPrev[1])** 2 <= pow(10, -6):
        #     print('reached a low weight diff')
        #     return weightsNew
        if iterations > 500:
            print('reached a 500 iterations')
            return weightsNew
        iterations += 1


weights = [0, -1]
weightsMinimized = minimize(weights, weights, .0001)
print(weightsMinimized)


def plotRegression(x):
    return weightsMinimized[0]+weightsMinimized[1]*x


regressionLine = numpy.array(range(-5, 60))
plt.scatter(inputX, inputy, c="red", alpha=.9, marker='o')
plt.axis([0, 100, 0, 100])
plt.plot(regressionLine, plotRegression(regressionLine))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
