import tensorflow as tf
from tensorflow import keras
from utils import *
from tqdm import tqdm
import sys


if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)


ALL_WORDS = ['right','eight','cat','tree','backward','learn','bed','happy','go','dog','no','wow','follow','nine','left','stop','three','sheila','one','bird','zero','seven','up','visual','marvin','two','house','down','six','yes','on','five','forward','off','four']
corrections = {word : word for word in ALL_WORDS + ['silence']}

if len(sys.argv) > 1:
	data = sys.argv[1:]
else:
	print('p3 predict.py [queries]')
	exit()
with tf.compat.v1.Session() as sess:
	tf.compat.v1.saved_model.load(sess, ['serve'], 'saved_model_v2')
	X = []
	for i in tqdm(range(len(data))):
		file = data[i]
		x, _ = parse_preprocess(file, None)
		X.append(sess.run(x))
	feed = {'IteratorGetNext:0': X, 'batch_size:0': len(data)}
	predictions = sess.run('output_batch:0', feed)
	corrections = autocorrect(predictions)
	for pred, corrected in zip(predictions, corrections):
		pred = dense_to_string(pred)
		print(pred, '--->', corrected)



