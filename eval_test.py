import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from utils import *
from model import *
from tqdm import tqdm
import sys, time, math
from scipy.io import wavfile

test_list, test_labels = read_list('train.txt', shuffle=True)
test_data = np.load('train_wav.npy')

model = CTC_Model()
model.build_data_pipeline([None, 16000], labels_shape=test_labels.shape, prefetch=0)
model.build_inference()
model.create_optimiser(0.1)

shuffled_labels = []

batch = 256
total = len(test_data)
with tf.Session() as sess:
	saver = tf.train.Saver()
	saver.restore(sess, 'saved_weights/weights-4098')
	feed_data = {'input_data:0': test_data, 'input_labels:0': test_labels, 'batch_size:0': batch}
	sess.run(model.init, feed_data)
	epoch_accuracy = 0
	predictions = []
	n_batch = 0
	num_batch = math.ceil(total / batch)
	while True:
		try:
			t = time.time()
			#pred = sess.run(model.decoded)
			pred, raw_accurary, labels = sess.run([model.decoded, model.accuracy, model._labels])
			t = time.time() - t
			n_batch += 1
			print('\rtesting... %.2f'%(n_batch/num_batch*100)+'%'+'(%.3fs)'%(t), end='')
			predictions.extend(pred)
			shuffled_labels.extend(labels)
			epoch_accuracy += raw_accurary
		except tf.errors.OutOfRangeError:
			break
	print()
	count = 0
	count2 = 0
	corrections, string_pred = autocorrect(predictions)
	for corrected, pred, truth in zip(corrections, string_pred, shuffled_labels):
		truth = dense_to_string(truth).strip('_')
		if np.random.random() < 0.01:
			print(pred, '<--->', corrected, '<--->', truth)
		if corrected == truth:
			count += 1
			count2 += 1
		elif corrected not in KEYWORDS and truth not in KEYWORDS:
			count2 += 1

	print('\nraw accuracy: ', epoch_accuracy / total)
	print('edited accuracy:', count / total)
	print('edited accuracy2:', count2 / total)





