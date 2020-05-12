import numpy as np
import random as rn
from scipy import io
import h5py
from scipy.misc import imresize as resize
import pdb

np.random.seed(42)
rn.seed(12345)

def computemAP(hammingRank, groundTruthSimilarity, trackPrec = False):
	[Q, N] = hammingRank.shape
	pos = np.arange(N)+1
	MAP = 0
	numSucc = 0
	if trackPrec:
		rareClsVsPrec = []
	for i in range(Q):
		ngb = groundTruthSimilarity[i, np.asarray(hammingRank[i,:], dtype='int32')]
		nRel = np.sum(ngb)
		if nRel > 0:
			prec = np.divide(np.cumsum(ngb), pos)
			prec = prec
			ap = np.mean(prec[np.asarray(ngb, dtype='bool')])
			MAP = MAP + ap
			numSucc = numSucc + 1
			if trackPrec:
				rareClsVsPrec.append((nRel, ap))
	MAP = float(MAP)/numSucc
	if trackPrec:
		import scipy.io as sio 
		sio.savemat('rareClsVsPrec.mat',{'rareClsVsPrec':rareClsVsPrec})
	return MAP


def computeSimilarity(queryLabels, databaseLabels, typeOfData='singleLabelled'):
	groundTruthSimilarityMatrix = np.zeros((queryLabels.shape[0], databaseLabels.shape[0]))
	if typeOfData=='singleLabelled':
		queryLabels = np.reshape(queryLabels, (max(queryLabels.shape),))
		databaseLabels = np.reshape(databaseLabels, (max(databaseLabels.shape),))
		for i in range(queryLabels.shape[0]):
			groundTruthSimilarityMatrix[i,:] = queryLabels[i] == databaseLabels
	elif typeOfData=='multiLabelled':
		for i in range(queryLabels.shape[0]):
			curQue = queryLabels[i][:]
			if sum(curQue) != 0:
				threshold = 1
				sim = np.sum(np.logical_and(curQue, databaseLabels), axis=-1)
				den = np.sum(np.logical_or(curQue, databaseLabels), axis=-1)
				groundTruthSimilarityMatrix[i][np.where(sim >= threshold)[0]] = 1
	groundTruthSimilarityMatrix = np.asarray(groundTruthSimilarityMatrix, dtype='float32')
	return groundTruthSimilarityMatrix


def calcHammingRank(queryHashes, databaseHashes):
	hammingDist = np.zeros((queryHashes.shape[0], databaseHashes.shape[0]))
	hammingRank = np.zeros((queryHashes.shape[0], databaseHashes.shape[0]))
	for i in range(queryHashes.shape[0]):
		hammingDist[i] = np.reshape(np.sum(np.abs(queryHashes[i] - databaseHashes), axis=1), (databaseHashes.shape[0], ))
		hammingRank[i] = np.argsort(hammingDist[i])
	return hammingDist, hammingRank


def prAtK(hammingDist, groundTruthSimilarity, k):
	countOrNot = np.array(hammingDist <= k, dtype='int32')
	newSim = np.multiply(groundTruthSimilarity, countOrNot)
	countOrNot = countOrNot + 0.000001
	#pdb.set_trace()
	prec = np.mean(np.divide(np.sum(newSim, axis=-1), np.sum(countOrNot, axis=-1)))
	rec = np.mean(np.divide(np.sum(newSim, axis=-1), np.sum(groundTruthSimilarity, axis=-1)))
	return (prec, rec)

def numUniqueHashes(x):
	y = np.unique(x, axis=0)
	return y.shape[0]

def getShannonEntropy(x):
	raise NotImplementedError

def getAvgHashHistogram(hammingDist, nBits=12):
	finalHist = np.zeros((nBits,))
	for i in range(hammingDist.shape[0]):
		# pdb.set_trace()
		finalHist = finalHist + np.histogram(hammingDist[i,:], nBits)[0]
	return finalHist

def getCosineSimilarity(x, batchSize=50, save=True, getFullMatrix=False):
	from scipy.spatial.distance import cdist
	distances = np.zeros((x.shape[0], x.shape[0]), dtype='float32')
	for i in range(int(x.shape[0]/batchSize)):
		for j in range(int(x.shape[0]/batchSize)):
			if np.sum(distances[j*batchSize:(j+1)*batchSize, i*batchSize:(i+1)*batchSize]) == 0:
				dst = cdist(x[i*batchSize:(i+1)*batchSize] , x[j*batchSize:(j+1)*batchSize] ,  'cosine')
				distances[i*batchSize:(i+1)*batchSize, j*batchSize:(j+1)*batchSize] = dst
			elif getFullMatrix:
				distances[i*batchSize:(i+1)*batchSize, j*batchSize:(j+1)*batchSize] = distances[j*batchSize:(j+1)*batchSize, i*batchSize:(i+1)*batchSize]
	return distances

def computeAccuracy(predictions, groundTruths):
	predictions = prepLabelData(predictions, sourceType='oneHot', targetType='uint')
	groundTruths = prepLabelData(groundTruths, sourceType='uint', targetType='uint')
	acc = np.sum((predictions == groundTruths))*100/predictions.shape[0]
	return acc