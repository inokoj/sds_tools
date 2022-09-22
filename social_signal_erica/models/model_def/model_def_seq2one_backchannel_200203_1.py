from model_def_default import ModelDefDefault

class ModelDef(ModelDefDefault):
	
	# 使用するモデルの種類
	model_type = 'seq2one'

	# 入力データの次元数
	input_dim = 40

	# バッチサイズ
	batch_size = 64

	# 使用するデータ
	name_data = 'filterbank'

	name_label = 'backchannel'
	task = 'bc'

	do_z_normalize = True
	label_balanced = True

	type_z_normalize = 'whole'

	use_middle_trained_model = False

	# 学習パラメータ
	optimizer = 'adam'
	learning_rate = 0.001
	weight_decay = 0.0001
	gradient_clip = 5

	num_epoch = 50

	test_start_epoch = 1
	test_end_epoch = 50

	# # 損失関数
	# loss_function = 'binary_cross_entroy_with_sigmoid'

	# # # 入力データの次元数
	# # input_dim = 40

	# Encoder (実際には２つ)
	encoder_num_layer = 2
	encoder_num_unit = 256
	encoder_ratio_dropout = 0.3
	encoder_is_bidirectional = True

	# Decoder
	decoder_num_layer = 0
	decoder_num_unit = 256
	decoder_ratio_dropout = 0.0
	decoder_dim_output = 2

	# # 途中までおこなった学習済みデータを使って学習を進める
	# use_middle_trained_model = False

	# # 何EPOCHずつモデルを保存するか
	# epoch_step_save_model = 5

	# # テストのパラメータ
	# test_start_epoch = 1
	# test_end_epoch = 50
	# test_batch_size = 512

	def __init__(self):
		
		self.make_filenames()

		# # 学習データセット全体の平均と分散を格納
		# self.filename_znormalize_output = self.data_dir + 'model/' + self.param_id + '/zscore'
