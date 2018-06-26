import numpy as np 
import h5py as hp 
import pdb
import sys
sys.path.insert(0, './utils')
import utils
import csv

resultsFolder="./../hashingBaselines/results/"
approaches = ["DSH_CVPR_2016_Caffe", "SSDH-PAMI", "ITQ", "SpH", "MLH"]
datasets = ["CIFAR-10", "NUS", "MIR"]
nBits = [12, 24, 32, 48]
nLevels = 3
toCompute = ['mAP', 'prec@0', 'rec@0', 'prec@1', 'rec@1', 'prec@2', 'rec@2', 'numUniqueHashes']
results = np.zeros((len(approaches), len(datasets), len(nBits), len(toCompute)))

# from scipy import io as sio

# data = sio.loadmat('/media/vijetha/DATA/vijetha/Dropbox (ASU)/CVPR_2018_multiHashing_dropbox/nus/NUSHashes16.mat')
# trainHashes = data["galleryHashes"]
# testHashes = data["queryHashes"]
# trainLabels = data["galleryCls"]
# testLabels = data["queryCls"]
# typeOfData = 'multiLabelled'
# groundTruthSimilarity = utils.computeSimilarity(testLabels, trainLabels, typeOfData)
# hammingDist, hammingRank = utils.calcHammingRank(testHashes, trainHashes)
# mAP=utils.computemAP(hammingRank, groundTruthSimilarity)
# curRes = [mAP]
# #pdb.set_trace()
# for l in range(nLevels):
# 	pre, rec = utils.prAtK(hammingDist, groundTruthSimilarity, l)
# 	curRes.append(pre)
# 	curRes.append(rec)
# uHashes = utils.numUniqueHashes(trainHashes)
# curRes.append(uHashes)
# results[0, 0 ,0, :] = curRes
# print(results)
# pdb.set_trace()
for i in range(len(approaches)):
	for j in range(len(datasets)):
		for p in range(len(nBits)):
			try:
				data = hp.File(resultsFolder+approaches[i]+'/'+datasets[j]+'/'+'codesAndLabels_'+str(nBits[p])+'.h5', 'r')
				trainHashes = data['trainHashes'][:]
				trainLabels = data['trainLabels'][:]
				testHashes = data['testHashes'][:]
				testLabels = data['testLabels'][:]
				trainHashes = utils.getRealValuedToCode(trainHashes, 0)
				testHashes = utils.getRealValuedToCode(testHashes, 0)
				if datasets[j] == 'CIFAR-10':
					typeOfData = 'singleLabelled'
				elif datasets[j] == 'NUS':
					typeOfData = 'multiLabelled'
				groundTruthSimilarity = utils.computeSimilarity(testLabels, trainLabels, typeOfData)
				hammingDist, hammingRank = utils.calcHammingRank(testHashes, trainHashes)
				mAP=utils.computemAP(hammingRank, groundTruthSimilarity)
				curRes = [mAP]
				#pdb.set_trace()
				for l in range(nLevels):
					pre, rec = utils.prAtK(hammingDist, groundTruthSimilarity, l)
					curRes.append(pre)
					curRes.append(rec)
				uHashes = utils.numUniqueHashes(trainHashes)
				curRes.append(uHashes)
			except:
				curRes = [-100]*(2*nLevels+2)
			results[i, j ,p, :] = curRes
utils.writeHashingResultsToCsv(results, 'PRresults.csv', 'w', approaches, datasets, nBits, toCompute)