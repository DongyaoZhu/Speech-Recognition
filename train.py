from model import CTC_Model
import tensorflow as tf    
from utils import *
import os, time, math
import numpy as np
from argparse import ArgumentParser


if tf.__version__ == '1.14.0':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)


parser = ArgumentParser()
parser.add_argument('train_path',
                    help='path to training data list',
                    metavar='TRAIN_PATH')
parser.add_argument('-noise',
                    dest='noise_path',
                    help='path to noise',
                    metavar='NOISE_PATH')
parser.add_argument('-val',
                    dest='val_path',
                    help='path to validation data list',
                    metavar='VALIDATION_PATH')
parser.add_argument('-test',
                    dest='test_path',
                    help='path to test data list',
                    metavar='TEST_PATH')
parser.add_argument('-e', default=1, type=int,
                    dest='num_epoch', metavar='NUM EPOCH',
                    help='number of epoch to train')
parser.add_argument('-p', default=0, type=int,
                    dest='prefetch', metavar='NUM PREFETCH',
                    help='number of batches to prefetch')
parser.add_argument('-b', default=128, type=int,
                    dest='batch', metavar='BATCH SIZE',
                    help='batch size')
parser.add_argument('-lr', default=0.001, type=float,
                    dest='lr', metavar='LEARNING RATE',
                    help='learning rate')
parser.add_argument('-r',
                    dest='restore_path', metavar='RESTORE PATH',
                    help='restore weights from specified path')
parser.add_argument('-s',
                    dest='save_path', metavar='SAVE PATH',
                    help='save weights to specified path')
args = parser.parse_args()


train_list, train_labels = read_list(args.train_path, shuffle=True)
tf.reset_default_graph()
model = CTC_Model()
model.build_data_pipeline([None, 16000], labels_shape=train_labels.shape, prefetch=args.prefetch)
model.build_inference()
model.create_optimiser(args.lr)


t = time.time()
train_data = np.load('train_wav.npy')
train_list, train_labels = read_list(args.train_path, shuffle=True)
print('done reading train data, used %.3fs' % (time.time() - t))


if args.noise_path and os.path.isfile(args.noise_path):
    t = time.time()
    noise_list, _ = read_list(args.noise_path, shuffle=True)
    noise_data = np.load('noise_wav.npy')
    print('done reading noise data, used %.3fs' % (time.time() - t))


if args.val_path and os.path.isfile(args.val_path):
    t = time.time()
    val_list, val_labels = read_list(args.val_path, shuffle=True)
    val_data = np.load('validation_wav.npy')
    print('done reading val data, used %.3fs' % (time.time() - t))


if args.test_path and os.path.isfile(args.test_path):
    t = time.time()
    test_list, test_labels = read_list(args.test_path, shuffle=True)
    test_data = np.load('test_wav.npy')
    print('done reading test data, used %.3fs' % (time.time() - t))


total = len(train_data)
num_batch = math.ceil(total / args.batch)


saver = tf.train.Saver()
# export_dir = 'saved_model_v2'
# builder = tf.compat.v1.saved_model.Builder(export_dir)
# signature = tf.compat.v1.saved_model.predict_signature_def(
#     inputs={'input_batch': model._input}, 
#     outputs={'output_batch': model.decoded})
#from tensorflow.core.protobuf import rewriter_config_pb2
#config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
#off = rewriter_config_pb2.RewriterConfig.OFF
#config_proto.graph_options.rewrite_options.arithmetic_optimization = off


with tf.Session() as sess:
    if args.restore_path:
        saver.restore(sess, args.restore_path)
    else:
        sess.run(tf.global_variables_initializer())
    # builder.add_meta_graph_and_variables(sess=sess,
    #                                     tags=['serve'],
    #                                     signature_def_map={'predict':signature})
    for epoch in range(1, args.num_epoch + 1):
        #feed = {'inputs/input_data:0': train_data, 
                #'inputs/input_labels:0': train_labels}
        if args.noise_path:
            indices = np.random.choice(range(len(noise_data)), size=total)
            feed = {'input_data:0': 0.99 * train_data + 0.01 * noise_data[indices],
                    'input_labels:0': train_labels,
                    'batch_size:0': args.batch}
        else:
            feed = {'input_data:0': train_data,
                    'input_labels:0': train_labels,
                    'batch_size:0': args.batch}
        sess.run(model.init, feed)
        epoch_loss, epoch_edit_dist, epoch_accuracy = 0, 0, 0
        step = 0
        while True:
            try:
                t = time.time()
                loss, _, decoded, edit_dist, accuracy, Y, gs = \
                    sess.run([model.loss, model.opt, model.decoded, \
                        model.edit_dist, model.accuracy, model._labels, model.global_step])
                t = time.time() - t
                step += 1
                progress = '\r[epoch %d step %d] %.2f' % (epoch, gs, step / num_batch * 100) + '%'
                seconds = (num_batch - step) * t
                minutes = seconds // 60
                hours = minutes // 60
                seconds %= 60
                loss_ = ' [loss %.3f]' % (loss / len(Y))
                timing = ' [BATCH %.3fs / ETA %dm %.3fs]     ' % (t, minutes, seconds)
                if hours > 0:
                    timing = ' [BATCH %.3fs / ETA %dh %dm %.3fs]     ' % (t, hours, minutes, seconds)
                print(progress + loss_ + timing, end='')

                epoch_loss += loss
                epoch_edit_dist += edit_dist
                epoch_accuracy += accuracy
            except tf.errors.OutOfRangeError:
                break

        print('\n  loss: %.3f' % (epoch_loss / total), end='')
        print('  edit_dist: %.3f' % (epoch_edit_dist / total))
        print('  accuracy: %.3f' % (epoch_accuracy / total))
        model.visualise(decoded, Y)
        if args.save_path:
            print('saving this run at %d' % gs)
            saver.save(sess, args.save_path, global_step=model.global_step)
        if args.val_path:
            model.test(sess, val_data, val_labels, batch=args.batch * 4)
    if args.test_path:
        model.test(sess, test_data, test_labels, batch=args.batch * 4)
    # builder.save()
    
