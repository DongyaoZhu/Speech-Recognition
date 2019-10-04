import os, sys, time
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import tensorflow as tf

VOCAB = [''] + [chr(ord('a') + i) for i in range(26)] + ['_']

MAX_LABEL = len(VOCAB) - 1

CHAR_TO_INDEX = dict(zip(VOCAB, range(len(VOCAB))))

KEYWORDS = {'silence', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'}

ALL_WORDS = ['right','eight','cat','tree','backward','learn','bed','happy','go','dog',\
	'no','wow','follow','nine','left','stop','three','sheila','one','bird','zero','seven',\
	'up','visual','marvin','two','house','down','six','yes','on','five','forward','off','four']

def string_to_dense(string, length=0, pad=0):
	if string == 'silence':
		return [0] * 8
	return [CHAR_TO_INDEX[char] for char in string] + length * [pad]
	
def dense_to_string(dense):
	if sum(dense) == 0:
		return 'silence'
	return ''.join([VOCAB[i] for i in dense])

def read_list(path, shuffle=False):
	data, labels = [], []
	with open(path) as file:
		line = file.readline()
		while line:
			label = line.split('/')[0]
			abs_name = 'data/' + line[:-1] 
			data.append(abs_name)
			labels.append(string_to_dense(label, length=8 - len(label), pad=MAX_LABEL))
			line = file.readline()
	print('list len:', len(data))
	if not shuffle:
		return data, np.asarray(labels, np.int32)
	np.random.seed(0)
	indices = np.random.permutation(range(len(data)))
	new_data, new_labels = [], []
	for i in indices:
		new_data.append(data[i])
		new_labels.append(labels[i])
	return new_data, np.asarray(new_labels, np.int32)

### correction on prediction using ED ###

def edit_distance(source, target):
	ls, lt = len(source) + 1, len(target) + 1
	k1, k2, k3 = 1.0, 1.0, 1.0
	ed = [[i * k1 for i in range(lt)]] + [[i*k1] * lt for i in range(1, ls)]
	for i in range(1, ls):
		for j in range(1, lt):
			if source[i - 1] == target[j - 1]:
				ed[i][j] = ed[i - 1][j - 1] 
			else:
				ed[i][j] = min([ed[i - 1][j]*k2, ed[i][j - 1]*k3, ed[i - 1][j - 1]]) + 1
	return ed[-1][-1]

def autocorrect(predictions):
	cached_corrections = {word: word for word in ALL_WORDS + ['silence']}
	cached_corrections['olow'] = 'follow'
	corrections = []
	string_pred = []
	for i, pred in enumerate(predictions):
		pred = dense_to_string(pred)
		string_pred.append(pred)
		min_dist = '', 100
		if pred in cached_corrections:
			corrected = cached_corrections[pred]
		else:
			for v in ALL_WORDS:
				e = edit_distance(pred, v)
				if e < min_dist[1]:
					min_dist = v, e
			corrected = min_dist[0]
			cached_corrections[pred] = corrected
		corrections.append(corrected)
	return corrections, string_pred

def generator(*args):
	def gen():
		for data in zip(*args):
			yield data
	return gen

