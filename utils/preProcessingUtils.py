import numpy as np
np.random.seed(42)
import random as rn
rn.seed(12345)

import pdb
from scipy import io
import h5py
# from skimage.transform import resize
from scipy.misc import imresize as resize



def getTriplets(nSamples, labels, batch_size):
	"""
	"""
	triplets = np.zeros((batch_size, 3), dtype='int32')
	ind = 0
	toFill = np.sum(np.sum(triplets, axis=-1)==0)
	while (toFill != 0):
		contextSamples = np.random.permutation(nSamples)
		contextSamples = contextSamples[0:batch_size]
		posSamples = np.random.permutation(nSamples)
		posSamples = posSamples[0:batch_size]
		negSamples = np.random.permutation(nSamples)
		negSamples = negSamples[0:batch_size]	
		L = labels[contextSamples]
		L_plus = labels[posSamples]
		L_minus = labels[negSamples]
		d_plus = np.reshape(np.sum(np.abs(L - L_plus), axis=1), (batch_size))
		d_minus = np.reshape(np.sum(np.abs(L - L_minus), axis=1), (batch_size))
		atleastOneCommonLabel_plus = np.sum(np.logical_and(L, L_plus), axis=1)
		atleastOneCommonLabel_minus = np.sum(np.logical_and(L, L_minus), axis=1)
		correct = np.logical_and(d_plus < d_minus, atleastOneCommonLabel_plus)
		reverse = np.logical_and(d_minus < d_plus, atleastOneCommonLabel_minus)
		nCorrect = np.sum(correct)
		nReverse = np.sum(reverse)
		triplets[ind:ind+nCorrect,0] = contextSamples[correct][0:toFill]
		triplets[ind:ind+nCorrect,1] = posSamples[correct][0:toFill]
		triplets[ind:ind+nCorrect,2] = negSamples[correct][0:toFill]
		ind = ind + nCorrect
		toFill = np.sum(np.sum(triplets, axis=-1)==0)
		triplets[ind:ind+nReverse,0] = contextSamples[reverse][0:toFill]
		triplets[ind:ind+nReverse,1] = negSamples[reverse][0:toFill]
		triplets[ind:ind+nReverse,2] = posSamples[reverse][0:toFill]
		ind = ind + nReverse
		toFill = np.sum(np.sum(triplets, axis=-1)==0)
	return triplets