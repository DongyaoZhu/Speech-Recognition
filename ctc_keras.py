
from utils import read_data, split_data
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Activation, Bidirectional, Dropout, \
	Embedding, Dense, GRU, Lambda, BatchNormalization, Flatten, Conv1D, MaxPooling1D
from keras.backend import ctc_label_dense_to_sparse, ctc_batch_cost, ctc_decode
from keras.metrics import categorical_accuracy, sparse_categorical_accuracy

tf.logging.set_verbosity(tf.logging.ERROR)
# '''
data, labels, X_length = read_data('spectrogram_no_pad', pad=1, shuffle=True)
val_split = 9984
test_split = val_split + 128 * 30
train_data, train_labels, train_length, val_data, val_labels, val_length, \
test_data, test_labels, test_length = split_data(data, labels, val_split, test_split, X_length)
print('done reading data')

def stream(data, labels, X_length):
    indices = list(range(len(data)))
    while True:
        for i in indices:
            yield data[i], labels[i], X_length[i]

def get_dense_batch(stream, batch_size):
    data_batch, label_batch, x_length, y_length = [[] for i in range(4)]
    while 1:
        for datum, label, length in stream:
            data_batch.append(datum)
            label_batch.append(label)
            x_length.append(length)
            y_length.append(len(label))
            if len(data_batch) == batch_size:
                input_={'data_batch': data_batch,
                        'label_batch:': label_batch,
                        'input_length': x_length,
                        'label_length': y_length}
                output_={'ctc': np.zeros([batch_size])}
                # print([(i, np.shape(input_[i])) for i in input_])
                # print(label_batch)
                yield ([data_batch, label_batch, x_length, y_length], [output_['ctc']])
                # yield (input_, output_)
                data_batch, label_batch, x_length, y_length = [[] for i in range(4)]
    yield data_batch, label_batch, x_length, y_length

batch_size = 128
train_stream = stream(train_data, train_labels, train_length)
train_generator = get_dense_batch(train_stream, batch_size)
val_stream = stream(val_data, val_labels, val_length)
val_generator = get_dense_batch(val_stream, len(val_data))
# '''
def ctc(args):
    a,b,c,d=args
    return ctc_batch_cost(a,b,c,d)

input_layer = Input(shape=(44, 32), name='data_batch')
conv1 = Conv1D(filters=32, kernel_size=2, activation='relu')(input_layer)
conv2 = Conv1D(filters=32, kernel_size=2, activation='relu')(conv1)
pool1 = MaxPooling1D(pool_size=2)(conv2)
# conv3 = Conv1D(filters=32, kernel_size=2, activation='relu')(pool1)
# conv4 = Conv1D(filters=32, kernel_size=2, activation='relu')(conv3)
# pool2 = MaxPooling1D(pool_size=2)(conv4)
rnn = GRU(32, return_sequences=True)(pool1)
batch_norm1 = BatchNormalization()(rnn)
activation = Activation('tanh')(batch_norm1)
dropout = Dropout(rate=0.5)(activation)
output = Dense(27, activation='softmax')(pool1)

label_layer = Input(shape=[8], name='label_batch')
print(label_layer.shape)
input_length = Input(shape=[1], name='input_length')
label_length = Input(shape=[1], name='label_length')
loss = Lambda(ctc, output_shape=(1,), name='ctc')\
    ([label_layer, output, input_length, label_length])

model = Model(inputs=[input_layer, label_layer, input_length, label_length], outputs=[loss])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, 
            optimizer=keras.optimizers.Adam(lr = 0.001, decay = 0.01))
model.summary()

history = model.fit_generator(generator=train_generator, 
    steps_per_epoch=78, 
    epochs=1, 
    # validation_data=val_generator,
    validation_steps=1)
# history = model.fit(train_data,
#                     train_labels,
#                     epochs=5,
#                     batch_size=128,
#                     # validation_data=(val_data, val_labels),
#                     verbose=1,
#                     shuffle=0)
# print('test:')
# model.evaluate(test_data, test_labels)

