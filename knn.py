#!/usr/bin/python3

import pickle, sys, operator
from sklearn.decomposition import PCA
from collections import defaultdict
import numpy as np

k=''
d=''
n=''
dataFile=''
totalImageCount = 1000	#defined in problem statement

def readInput():
    global k,d,n,dataFile
    k = int(sys.argv[1])
    d = int(sys.argv[2])
    n = int(sys.argv[3])
    dataFile = sys.argv[4]

def unpickleData(fileName):
    with open(fileName, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

def splitImageData(data, n):
    global totalImageCount
    imageData = data[b'data'][:totalImageCount]
    labelData = data[b'labels'][:totalImageCount]
    testImages = np.array(imageData[:n])	#First N images should be used as test data
    testLabels = labelData[:n]
    trainImages = np.array(imageData[n:])	#remaining images (out of 1000) to be used as training data
    trainLabels = labelData[n:]
    return trainImages, trainLabels, testImages, testLabels

def convertToGrayscale(rgbImages):
    rgbImagesReshaped = np.reshape(rgbImages, (-1,3,1024))
    return np.dot([0.299, 0.587, 0.114], rgbImagesReshaped)

def dimensionalityReduction(trainImages, testImages, dim):
    p = PCA(n_components = d, svd_solver = 'full').fit(trainImages)
    trainImages = p.transform(trainImages)
    testImages = p.transform(testImages)
    return trainImages, testImages

def classifyKNN(img, trainImages, trainLabels):
    global k
    distances = np.sqrt(np.sum(np.square(trainImages - img), axis = 1))		#smaller value => closer to test item
    invertDistances = 1./distances
    disDict = {key:v for key,v in enumerate(invertDistances)}
    maxNeighbors = sorted(disDict.items(), key=operator.itemgetter(1))[-k:]	#sort the dictionary and pick the last k elements
    votes = defaultdict(float)
    for v in maxNeighbors:
        votes[trainLabels[v[0]]] += v[1]
    return max(votes.items(), key=operator.itemgetter(1))[0]

def predict(trainImages, trainLabels, testImages):
    prediction = []
    for i in range(len(testImages)):
        predictedClass = classifyKNN(testImages[i], trainImages, trainLabels)
        prediction.append(predictedClass)
    return prediction

def writeOutput(prediction, groundTruth):
    outputFile = open('output.txt', 'w')
    for i in range(len(prediction)):
        outputFile.write(str(prediction[i])+" "+str(groundTruth[i])+"\n")
    outputFile.close()

def main():
    global k,d,n,dataFile
    readInput()
    data = unpickleData(dataFile)
    trainImages, trainLabels, testImages, testLabels = splitImageData(data, n)
    trainImages = convertToGrayscale(trainImages)
    testImages = convertToGrayscale(testImages)
    trainImages, testImages = dimensionalityReduction(trainImages, testImages, d)
    prediction = predict(trainImages, trainLabels, testImages)
    writeOutput(prediction, testLabels)

main()
