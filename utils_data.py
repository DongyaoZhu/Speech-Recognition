import tensorflow as tf


def create_data_pipeline(data_shape, labels_shape=None, prefetch=4):
    batch_size = tf.placeholder_with_default(tf.constant(128, dtype=tf.int64), [], name='batch_size')
    dp = tf.placeholder(tf.float32, [None, *data_shape[1:]], name='input_data')
    if labels_shape is not None:
        lp = tf.placeholder(tf.int32, [None, *labels_shape[1:]], name='input_labels')
        dataset = tf.data.Dataset.from_tensor_slices((dp, lp)).shuffle(20000).batch(batch_size, drop_remainder=False)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(dp).shuffle(20000).batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(prefetch)
    iterator = dataset.make_initializable_iterator()
    init = iterator.initializer
    batch_wav, _labels = iterator.get_next()
    _spectrogram = preprocess(batch_wav)
    return _spectrogram, _labels, init


def preprocess(batch_wav):
    frame_length = 480 
    frame_step = 160
    num_mel_bins= 46
    sr = 16000
    stft = tf.abs(tf.contrib.signal.stft(signals=batch_wav, 
                                frame_length=frame_length,
                                frame_step=frame_step,
                                fft_length=frame_length,
                                window_fn=tf.contrib.signal.hann_window,
                                pad_end=False))
    #num_spectrogram_bins = 321
    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
                                num_mel_bins=num_mel_bins,
                                num_spectrogram_bins=num_spectrogram_bins,
                                sample_rate=sr,
                                lower_edge_hertz=20.0,
                                upper_edge_hertz=4000.0,
                                dtype=tf.float32)
    feature = tf.tensordot(stft, linear_to_mel_weight_matrix, 1)
    feature.set_shape(stft.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # feature = tf.math.log(feature + 1e-6)
    # feature = tf.contrib.signal.mfccs_from_log_mel_spectrograms(feature)[..., :13]
    # feature = tf.expand_dims(feature, axis=-1)
    # return feature
    v_max = tf.reduce_max(feature, axis=[1, 2], keep_dims=True)
    v_min = tf.reduce_min(feature, axis=[1, 2], keep_dims=True)
    is_zero = tf.to_float(tf.equal(v_max - v_min, 0))
    feature = (feature - v_min) / (v_max - v_min + is_zero)

    epsilon = 0.001
    feature = tf.log(feature + epsilon)
    v_min = tf.log(epsilon)
    v_max = tf.log(epsilon + 1)
    feature = (feature - v_min) / (v_max - v_min)
    feature = tf.expand_dims(feature, axis=-1)
    return feature


