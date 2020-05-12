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

def find_union_nms_bw_two_sets_of_boxes(boxes_a, boxes_b):
    new_boxes_a = [[b[0]+1.0, b[1]] for b in boxes_a]
    new_boxes_b = boxes_b
    m = len(new_boxes_a)
    n = len(new_boxes_b)
    new_boxes = new_boxes_a + new_boxes_b
    result = non_max_suppression(new_boxes, 0.4)
    if len(result) > m+n or len(result) < max(m, n):
        print("Bug in find_union_nms_bw_two_sets_of_boxes!")
        import pdb
        pdb.set_trace()
    ret_val = []
    for i in range(len(result)):
        if result[i][0] > 1.0:
            ret_val.append(['set_A', result[i][0]-1.0, result[i][1]])
        else:
            ret_val.append(['set_B', result[i][0], result[i][1]])
    return ret_val

def non_max_suppression(boxes, threshold=0.3):
    if len(boxes)==0:
        return []
    new_boxes = [(b[1][0], b[1][1], b[1][2], b[1][3]) for b in boxes]
    scores = [b[0] for b in boxes]
    scores = np.asarray(scores)
    ixs = list(scores.argsort()[::-1])
    pick = []
    while len(ixs)>0:
        b1 = new_boxes[ixs[0]]
        pick.append(ixs[0])
        del ixs[0]
        other_boxes = deepcopy(ixs)
        while len(other_boxes) > 0:
            b2 = new_boxes[other_boxes[0]]
            iou = per_object_iou(b1, b2)
            if iou > threshold:
                ixs.remove(other_boxes[0])
            del other_boxes[0]
    final_boxes = [[scores[i], new_boxes[i]] for i in pick]
    return final_boxes

def per_object_iou(boxA, boxB):
    boxA = (boxA[0], boxA[1], boxA[0]+boxA[2], boxA[1]+boxA[3])
    boxB = (boxB[0], boxB[1], boxB[0]+boxB[2], boxB[1]+boxB[3])
    if boxes_intersect(boxA, boxB) is False:
        return 0
    interArea = get_intersection_area(boxA, boxB)
    union = get_union_areas(boxA, boxB, interArea=interArea)
    iou = interArea / union
    assert iou >= 0
    return iou
    
def boxes_intersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True
    
def get_intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)
    
def get_union_areas(boxA, boxB, interArea=None):
    area_A = get_area(boxA)
    area_B = get_area(boxB)
    if interArea is None:
        interArea = get_intersection_area(boxA, boxB)
    return float(area_A + area_B - interArea)

def get_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)