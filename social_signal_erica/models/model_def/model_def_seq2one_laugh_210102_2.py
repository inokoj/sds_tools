from model_def_default import ModelDefDefault

class ModelDef(ModelDefDefault):
	
	name_label = 'laugh'
	task = 'laugh'

	model_type = 'seq2one_gru'
	batch_size = 64
	
	do_z_normalize = True
	type_z_normalize = 'whole'

	label_balanced = True

	frame_stack = 1
	input_dim = 40 * frame_stack

	spec_augmentation = True
	do_oversampling = False

	#cnn_num_filters = 32
	#input_dim = 40 * frame_stack
	#cnn_kernel_size = 3
	#cnn_pooling_size = 3

	test_batch_size = 32

	use_middle_trained_model = False

	def __init__(self):
		self.make_filenames()