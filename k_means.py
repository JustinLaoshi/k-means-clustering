from __future__ import division
import numpy as np
import scipy.io
import math
import random
import sklearn
import itertools
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt 


def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = np.sum(errors) / float(true_labels.shape[0])
    indices = errors.nonzero()
    return err_rate, indices

def main():
	problem1()

def problem1():
	images = scipy.io.loadmat('data/mnist_data/images.mat')['images']
	images = images.transpose(2, 0, 1).reshape(60000, -1)
	kList = [5, 10, 20]

	def firstIndex(x):
		return x[1]

	def getClusterCenters(images, kList):
		retList = []
		for k in kList:
			print('on this k:', k)
			oldMean = images[np.random.choice(images.shape[0], k, replace=False),:]
			trueMean = images[np.random.choice(images.shape[0], k, replace=False),:]
			while not sameMean(trueMean, oldMean):
				oldMean = trueMean
				clusters = clusterImages(images, trueMean)
				trueMean = findNewMean(oldMean, clusters)
			retList.append((trueMean, clusters))
		return retList

	def clusterImages(images, mean):
		clusterDict  = {}
		for x in images:
			temp = []
			for i in enumerate(mean):
				norm = np.linalg.norm(x-mean[i[0]])
				temp.append((i[0], norm))
			best = min(temp, key=firstIndex)[0]
			try:
				clusterDict[best].append(x)
			except KeyError:
				clusterDict[best] = [x]
		return clusterDict

	def sameMean(trueMean, oldMean):
		oldList, newList = [tuple(i) for i in oldMean], [tuple(i) for i in trueMean]
		oldMeanSet, trueMeanSet = set(oldList), set(newList)
		return oldMeanSet == trueMeanSet	

	def findNewMean(oldMean, clusters):
	    clusterKeys = sorted(clusters.keys())
	    newMean = []
	    for k in clusterKeys:
	        newMean.append(np.mean(clusters[k], axis=0, dtype=None, out=None, keepdims=False))
	    return newMean
	
	def showClusters():
		retList = getClusterCenters(images, kList)
		for each in retList:
			mean = each[0]
			for m in mean:
				m = m.reshape(28, 28)
				plt.imshow(m)
				plt.show()


	showClusters()
