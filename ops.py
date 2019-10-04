import tensorflow as tf


def causal_conv(net, filters, kernel_size, dilation_rate, activation=None, padding='causal'):
    net = tf.keras.layers.Conv1D(filters=filters,
							kernel_size=kernel_size, 
							activation=activation,
							dilation_rate=dilation_rate,
							padding=padding)(net)   
    valid_causal_start = (kernel_size - 1) * dilation_rate
    return net[:, valid_causal_start:, :]


def gated_cnn(net, filters, kernel_size, dilation_rate):
	tanh = causal_conv(net, filters, kernel_size, dilation_rate, 'tanh', 'causal')
	sigmoid = causal_conv(net, filters, kernel_size, dilation_rate, 'sigmoid', 'causal')
	gated = tanh * sigmoid
	return gated


def residual_block(net, filters, kernel_size, dilation_rate, skip_filters, residual_filters, output_width=0):
	gated = gated_cnn(net, filters, kernel_size, dilation_rate)
	gated = tf.keras.layers.AveragePooling1D()(gated)
	skip_connection = causal_conv(gated, skip_filters, 1, 1, None, 'same')
	skip_connection = skip_connection[:, -output_width:, :]
	residual_connection = causal_conv(gated, residual_filters, 1, 1, None, 'same')
	short = net.shape[1] - residual_connection.shape[1]
	return skip_connection, net[:, short:, :] + residual_connection

'''unit tests
import numpy as np
a = tf.constant(np.arange(1, 16001).reshape([1,16000,1]), dtype=tf.float32)
b, c = residual_block(a, 1, 32, 2, 512, 64, 48)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a).shape)
    b, c = sess.run([b, c])
    print(b.shape)
    print(c.shape)
'''




