import numpy as np 
import h5py as hp 
import pdb
import sys
sys.path.insert(0, './utils')
import utils_training as ut
import utils_metrics as um
import utils_data as ud
import csv

resultsFolder="./../hashingBaselines/results/"
approaches = ["SSDH_PAMI_2017", "DSH_CVPR_2016_Caffe", "AGH", "DSH", "ITQ", "LSH", "PCAH", "UDHT"]
datasets = ["CIFAR-10", "NUS", "MIR"]
nBits = [12, 24, 32, 48]
nLevels = 3
toCompute = ['mAP', 'prec@0', 'rec@0', 'prec@1', 'rec@1', 'prec@2', 'rec@2', 'numUniqueHashes']
results = np.zeros((len(approaches), len(datasets), len(nBits), len(toCompute)))

for i in range(len(approaches)):
	for j in range(len(datasets)):
		for p in range(len(nBits)):
			try:
				print(str(approaches[i])+"--"+str(datasets[j])+"--"+str(nBits[p]))
				data = hp.File(resultsFolder+approaches[i]+'/'+datasets[j]+'/'+'codesAndLabels_'+str(nBits[p])+'.h5', 'r')
				ds = [key for key in data.keys() if '__' not in key]
				trainHashes = data['train_hashes'][:]
				testHashes = data['test_hashes'][:]
				if len(ds) == 4:
					trainLabels = data['train_labels'][:]
					testLabels = data['test_labels'][:]
					if datasets[j] == 'CIFAR-10':
						typeOfData = 'singleLabelled'
					elif datasets[j] == 'NUS':
						typeOfData = 'multiLabelled'
					groundTruthSimilarity = um.computeSimilarity(testLabels, trainLabels, typeOfData)
				elif len(ds) == 3:
					groundTruthSimilarity = data['groundTruthSimilarity'][:]
				if approaches == 'DSH_CVPR_2016_Caffe':
					trainHashes = ud.getRealValuedToCode(trainHashes, 0)
					testHashes = ud.getRealValuedToCode(testHashes, 0)
				hammingDist, hammingRank = um.calcHammingRank(testHashes, trainHashes)
				mAP=um.computemAP(hammingRank, groundTruthSimilarity, trackPrec=True)
				curRes = [mAP]
				for l in range(nLevels):
					pre, rec = um.prAtK(hammingDist, groundTruthSimilarity, l)
					curRes.append(pre)
					curRes.append(rec)
				allHashes = np.concatenate((trainHashes, testHashes), axis=0)
				uHashes = um.numUniqueHashes(allHashes)
				hashHist = um.getAvgHashHistogram(hammingDist) 
				print(hashHist)
				curRes.append(uHashes)
			except:
				curRes = [-100]*(2*nLevels+2)
			results[i, j ,p, :] = curRes
ut.writeHashingResultsToCsv(results, 'PRresults.csv', 'w', approaches, datasets, nBits, toCompute)