
NUM_CLASSES = 28

SKIP_FILTERS = 128

RES_FILTERS = 64

model_parameters = {

'num_classes': NUM_CLASSES,

'num_samples': 16000,

#'dilation_rates': [2 ** i for i in range(10)] * 5,
'dilation_rates': [1, 2, 4, 8, 16, 32, 64, 1, 2],

'preprocess':
{'filters': RES_FILTERS, 'kernel_size': 32, 'dilation_rate': 1, 'padding': 'causal'},

'postprocess1':
{'filters': SKIP_FILTERS, 'kernel_size': 1, 'dilation_rate': 1, 'padding': 'same'},

'postprocess2':
{'filters': NUM_CLASSES, 'kernel_size': 1, 'dilation_rate': 1, 'padding': 'same'},

}

for layer_id, dr in enumerate(model_parameters['dilation_rates']):
	params = {}
	params['filters'] = RES_FILTERS
	params['kernel_size'] = 2
	params['dilation_rate'] = dr
	params['skip_filters'] = SKIP_FILTERS
	params['residual_filters'] = RES_FILTERS
	#params['output_width'] = model_parameters['num_samples'] - 31 - sum(model_parameters['dilation_rates'])
	params['output_width'] = 19
	model_parameters['layer_%d' % (layer_id + 1)] = params

