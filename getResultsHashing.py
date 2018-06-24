import numpy as np 
import h5py as hp 
import pdb
import sys
sys.path.insert(0, './utils')
import utils
import csv

resultsFolder="./../hashingBaselines/results/"
approaches = ["DSH_CVPR_2016_Caffe"]
datasets = ["CIFAR-10", "NUS-WIDE"]
nBits = [12]
nLevels = 3
toCompute = ['mAP', 'prec@0', 'rec@0', 'prec@1', 'rec@1', 'prec@2', 'rec@2']
results = np.zeros((len(approaches), len(datasets), len(nBits), len(toCompute)))

# def getMAPResults():
# 	utils.writeCSVHeader('MAPresults.csv', datasets, nBits)
# 	with open('MAPresults.csv', 'a') as csvfile:
# 		mywriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
# 		for i in range(len(approaches)):
# 			row = [approaches[i]]
# 			for j in range(len(datasets)):
# 				for k in range(len(nBits)):
# 					try:
# 						data = hp.File(resultsFolder+approaches[i]+'/'+datasets[j]+'/'+'codesAndLabels_'+str(nBits[k])+'.h5', 'r')
# 						trainHashes = data['trainHashes'][:]
# 						trainLabels = data['trainLabels'][:]
# 						testHashes = data['testHashes'][:]
# 						testLabels = data['testLabels'][:]
# 						if datasets[j] == 'CIFAR-10':
# 							typeOfData = 'singleLabelled'
# 						elif datasets[j] == 'NUS-WIDE':
# 							typeOfData = 'multiLabelled'
# 						groundTruthSimilarity = utils.computeSimilarity(testLabels, trainLabels, typeOfData)
# 						mAP=utils.computemAP(testHashes, trainHashes, groundTruthSimilarity)
# 						row.append(mAP)
# 					except:
# 						row.append('N/A')
# 			mywriter.writerow(row)

# def getPRAtHammingDist(k, mode):
# 	utils.writeCSVHeader('PRresults.csv', datasets, nBits)
# 	with open('PRresults.csv', 'a') as csvfile:
# 		mywriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
# 		for i in range(len(approaches)):
# 			row = [approaches[i]]
# 			for j in range(len(datasets)):
# 				for p in range(len(nBits)):
# 					try:
# 						data = hp.File(resultsFolder+approaches[i]+'/'+datasets[j]+'/'+'codesAndLabels_'+str(nBits[p])+'.h5', 'r')
# 						trainHashes = data['trainHashes'][:]
# 						trainLabels = data['trainLabels'][:]
# 						testHashes = data['testHashes'][:]
# 						testLabels = data['testLabels'][:]
# 						if datasets[j] == 'CIFAR-10':
# 							typeOfData = 'singleLabelled'
# 						elif datasets[j] == 'NUS-WIDE':
# 							typeOfData = 'multiLabelled'
# 						groundTruthSimilarity = utils.computeSimilarity(testLabels, trainLabels, typeOfData)
# 						pdb.set_trace()
# 						pre, rec = utils.prAtK(testHashes, trainHashes, groundTruthSimilarity, k)
# 						# mAP=utils.computemAP(testHashes, trainHashes, groundTruthSimilarity)
# 						row.append([pre, rec])
# 					except:
# 						row.append('N/A')
# 			mywriter.writerow(row)
# # getMAPResults()
# getPRAtHammingDist(0, 1)

for i in range(len(approaches)):
	for j in range(len(datasets)):
		for p in range(len(nBits)):
			try:
				data = hp.File(resultsFolder+approaches[i]+'/'+datasets[j]+'/'+'codesAndLabels_'+str(nBits[p])+'.h5', 'r')
				trainHashes = data['trainHashes'][:]
				trainLabels = data['trainLabels'][:]
				testHashes = data['testHashes'][:]
				testLabels = data['testLabels'][:]
				if datasets[j] == 'CIFAR-10':
					typeOfData = 'singleLabelled'
				elif datasets[j] == 'NUS-WIDE':
					typeOfData = 'multiLabelled'
				groundTruthSimilarity = utils.computeSimilarity(testLabels, trainLabels, typeOfData)
				hammingDist, hammingRank = utils.calcHammingRank(testHashes, trainHashes)
				mAP=utils.computemAP(hammingRank, groundTruthSimilarity)
				curRes = [mAP]
				for l in range(nLevels):
					pre, rec = utils.prAtK(hammingDist, hammingRank, groundTruthSimilarity, l)
					curRes.append(pre)
					curRes.append(rec)
			except:
				curRes = [-100]*(2*nLevels+1)
			results[i, j ,p, :] = curRes
pdb.set_trace()