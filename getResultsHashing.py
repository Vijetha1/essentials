import numpy as np 
import h5py as hp 
import pdb
import sys
sys.path.insert(0, './utils')
import utils
import csv

resultsFolder="./../hashingBaselines/results/"
# approaches = ["DSH_CVPR_2016_Caffe", "SSDH_PAMI_2017", "ITQ", "SpH", "MLH"]
# approaches = ["SSDH_PAMI_2017", "DSH_CVPR_2016_Caffe", "AGH", "DSH", "ITQ", "LSH", "PCAH", "UDHT"]
approaches = ["UDHT"]
datasets = ["CIFAR-10", "NUS", "MIR"]
# datasets = ["NUS"]
nBits = [12, 24, 32, 48]
nLevels = 3
toCompute = ['mAP', 'prec@0', 'rec@0', 'prec@1', 'rec@1', 'prec@2', 'rec@2', 'numUniqueHashes']
results = np.zeros((len(approaches), len(datasets), len(nBits), len(toCompute)))

for i in range(len(approaches)):
	for j in range(len(datasets)):
		for p in range(len(nBits)):
			try:
				print(str(approaches[i])+"--"+str(datasets[j])+"--"+str(nBits[p]))
				# pdb.set_trace()
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
					groundTruthSimilarity = utils.computeSimilarity(testLabels, trainLabels, typeOfData)
				elif len(ds) == 3:
					groundTruthSimilarity = data['groundTruthSimilarity'][:]
				if approaches == 'DSH_CVPR_2016_Caffe':
					trainHashes = utils.getRealValuedToCode(trainHashes, 0)
					testHashes = utils.getRealValuedToCode(testHashes, 0)
				hammingDist, hammingRank = utils.calcHammingRank(testHashes, trainHashes)
				mAP=utils.computemAP(hammingRank, groundTruthSimilarity, trackPrec=True)
				curRes = [mAP]
				#pdb.set_trace()
				for l in range(nLevels):
					pre, rec = utils.prAtK(hammingDist, groundTruthSimilarity, l)
					curRes.append(pre)
					curRes.append(rec)
				allHashes = np.concatenate((trainHashes, testHashes), axis=0)
				uHashes = utils.numUniqueHashes(allHashes)
				hashHist = utils.getAvgHashHistogram(hammingDist) 
				# [ 9151543. 11422947. 11906292. 12862186. 19113238. 18151812. 25886490. 18146234. 21562496. 23541410. 20030515.  8224837.]
				print(hashHist)
				curRes.append(uHashes)
			except:
				curRes = [-100]*(2*nLevels+2)
			results[i, j ,p, :] = curRes
utils.writeHashingResultsToCsv(results, 'PRresults.csv', 'w', approaches, datasets, nBits, toCompute)