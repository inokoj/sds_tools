from model_def_seq2one_backchannel_200203_1 import ModelDef as ModelDefParent

class ModelDef(ModelDefParent):
	
	batch_size = 64
	
	do_z_normalize = True
	type_z_normalize = 'whole'

	label_balanced = True

	frame_stack = 1
	#input_dim = 40 * frame_stack

	spec_augmentation = True

	use_middle_trained_model = False

	def __init__(self):
		self.make_filenames()