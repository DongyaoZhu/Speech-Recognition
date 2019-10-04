from utils import *
from utils_data import *
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras


class CTC_Model():


    def __init__(self):
        self.is_training = tf.placeholder_with_default(True, shape=())
        self.num_classes = 28
        self.init = None


    def conv_bn(self, net, filters, kernel_size, padding='same', activation=True):
        conv1 = keras.layers.Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    padding=padding)(net)
        bn = keras.layers.BatchNormalization()(conv1)
        if activation:
            bn = keras.activations.relu(bn)
        return bn


    def residual(self, net, res_kernel):
        conv_bn_1 = self.conv_bn(net, *res_kernel)
        conv_bn_2 = self.conv_bn(conv_bn_1, *res_kernel, activation=False)
        res = keras.layers.Add()([net, conv_bn_2])
        relu = keras.activations.relu(res)
        return relu


    def conv_res_pool(self, net, conv_kernel, res_kernel, pool_kernel):
        net = self.conv_bn(net, *conv_kernel, padding='valid')
        net = self.residual(net, res_kernel)
        net = self.residual(net, res_kernel)
        if pool_kernel is not None:
            net = keras.layers.MaxPooling2D(*pool_kernel, padding='same')(net)
        net = keras.layers.BatchNormalization()(net)
        return net


    def create_cnn(self, net, conv_kernels, res_kernels, pool_kernels):
        for i in range(len(conv_kernels)):
            c, r, p = conv_kernels[i], res_kernels[i], pool_kernels[i]
            net = self.conv_res_pool(net, c, r, p)
        shape = net.shape
        net = tf.reshape(net, [-1, shape[1] * shape[2], shape[3]])
        # net = tf.reshape(net, [-1, shape[1], shape[2] * shape[3]])
        return net


    def gru_bn(self, net, hidden_sizes):
        cells = [keras.layers.GRUCell(size, dropout=0.5) for size in hidden_sizes]
        rnn = keras.layers.RNN(cell=cells, 
                               return_sequences=True,
                               go_backwards=False)(net, training=self.is_training)
        bn = keras.layers.BatchNormalization()(rnn)
        return bn


    def fc_bn_relu(self, net, units, leaky_relu=True):
        net = keras.layers.Dense(units=units)(net)
        net = keras.layers.BatchNormalization()(net)
        if leaky_relu:
            net = keras.layers.LeakyReLU(alpha=0.75)(net)
        return net


    def create_fc(self, net, units):
        for unit in units[:-1]:
            net = self.fc_bn_relu(net, unit, leaky_relu=True)
            net = self.add_dropout(net, 0.5)
        net = self.fc_bn_relu(net, units[-1], leaky_relu=False)
        return net


    def add_dropout(self, net, rate):
        net = keras.layers.Dropout(rate)(net, training=self.is_training)
        return net


    def build_data_pipeline(self, data_shape, **kwargs):
        self._spectrogram, self._labels, self.init = create_data_pipeline(data_shape, **kwargs)


    def build_inference(self):

        # conv/res kernel: [out_channel, kernel_size]
        conv_kernel1 = [128, (5,3)]
        res_kernel1 = [128, (3,3)]
        pool_kernel1 = [2]

        conv_kernel2 = [128, (3,3)]
        res_kernel2 = [128, (3,3)]
        pool_kernel2 = [2]

        conv_kernel3 = [256, (3,3)]
        res_kernel3 = [256, (3,3)]
        pool_kernel3 = [2]

        conv_kernels = [conv_kernel1, conv_kernel2, conv_kernel3]
        res_kernels = [res_kernel1, res_kernel2, res_kernel3]
        pool_kernels = [pool_kernel1, pool_kernel2, pool_kernel3]

        print('input shape:', self._spectrogram.shape)
        input_layer = keras.layers.BatchNormalization()(self._spectrogram)
        cnn_output = self.create_cnn(input_layer, conv_kernels, res_kernels, pool_kernels)
        print('cnn output shape:', cnn_output.shape)

        rnn_output = self.gru_bn(cnn_output, hidden_sizes=[768])
        print('rnn output shape:', rnn_output.shape)
        rnn_output = self.add_dropout(rnn_output, 0.5)

        fc_output = self.create_fc(rnn_output, units=[160, self.num_classes])
        print('fc output shape:', fc_output.shape)
        seq_len = fc_output.shape[1]

        self.logits = tf.transpose(fc_output, (1,0,2))
        self.sequence_length = tf.fill([tf.shape(self._spectrogram)[0]], 8)
        # self.sequence_length = tf.fill([tf.shape(self._spectrogram)[0]], seq_len)
        self.decode()


    def create_optimiser(self, start_lr):
        self.get_ctc_loss()
        self.get_optimiser(start_lr)
        self.eval_ctc()


    def get_ctc_loss(self):
        indices = tf.where(tf.not_equal(self._labels, self.num_classes - 1))
        values = tf.gather_nd(self._labels, indices)
        dense_shape = tf.shape(self._labels, out_type=tf.int64)
        self.sparse_labels = tf.SparseTensor(indices, values, dense_shape)
        ctc_loss = tf.nn.ctc_loss(
                        labels=self.sparse_labels,
                        inputs=self.logits,
                        sequence_length=self.sequence_length,
                        # sequence_length=self.sequence_length,
                        preprocess_collapse_repeated=True,
                        time_major=True)
        self.loss = tf.reduce_sum(ctc_loss)


    def get_optimiser(self, start_lr):
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
                            learning_rate=start_lr,
                            global_step=self.global_step,
                            decay_steps=4000,
                            decay_rate=0.96)
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, 
                                                            global_step=self.global_step)


    def decode(self):
        top_decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
                                        inputs=self.logits,
                                        sequence_length=self.sequence_length,
                                        beam_width=self.num_classes)
        self.sparse_decoded = tf.cast(top_decoded[0], dtype=tf.int32) # is sparse tensor
        self.decoded = tf.sparse.to_dense(self.sparse_decoded, name='output_batch')


    def eval_ctc(self):
        raw = tf.edit_distance(self.sparse_decoded, self.sparse_labels, normalize=False)
        ignore_inf = tf.where(tf.not_equal(raw, math.inf))
        no_inf = tf.gather_nd(raw, ignore_inf)
        self.edit_dist = tf.reduce_sum(no_inf)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(raw, 0), tf.int32))


    def test(self, sess, data, labels, batch=128):
        sess.run(self.init, {'input_data:0': data, 'input_labels:0': labels, 'batch_size:0': batch})
        total = len(data)

        num_batch = math.ceil(total / batch)
        n_batch = 0
        epoch_loss, epoch_accuracy, epoch_edit_dist = 0, 0, 0
        while True:
            try:
                _, decoded, edit_dist, accuracy, Y = sess.run([self.logits, self.decoded, \
                    self.edit_dist, self.accuracy, self._labels], {self.is_training: False})
                n_batch += 1
                print('\rtesting progress: %.2f'%((n_batch) / num_batch * 100) + '%', end='')
                epoch_edit_dist += edit_dist
                epoch_accuracy += accuracy
            except tf.errors.OutOfRangeError:
                break
        print('test edit distance: %.3f' % (epoch_edit_dist / total))
        print('test accuracy: %.3f' % (epoch_accuracy / total))
        self.visualise(decoded, Y) 
        return epoch_accuracy / total


    def visualise(self, decoded, Y):
        #np.random.seed(0)
        total = len(decoded)
        num_choices = total if total < 10 else 10
        for i in np.random.choice(range(total), num_choices, replace=False):
            print('  ['+dense_to_string(decoded[i])+'] <Pred-----Truth> ['+\
                dense_to_string(Y[i]).strip('_')+']')


