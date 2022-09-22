# パラメータの規定値

class ModelDefDefault(object):

	# 使用するモデルの種類
	model_type = 'seq2one'

	# input_dim: 120
	batch_size = 16

	# 使用するデータ
	name_data = 'filterbank'
	name_label = 'backchannel'
	do_z_normalize = True
	type_z_normalize = 'each'	# each / whole

	# 学習パラメータ
	optimizer = 'adam'
	learning_rate = 0.001
	weight_decay = 0.0001
	gradient_clip = 5
	num_epoch = 50

	# 損失関数
	loss_function = 'binary_cross_entropy_with_sigmoid'

	classification_type = 'binary'

	# 入力データの次元数
	input_dim = 40

	# フレームスタッキング
	frame_stack = 1

	# 出力データのクラス数
	output_dim = 2

	# Spec augmentation
	spec_augmentation = False

	# Oversampling
	do_oversampling = False

	# Encoder
	encoder_num_layer = 2
	encoder_num_unit = 256
	encoder_ratio_dropout = 0.3
	encoder_is_bidirectional = True

	# Decoder
	decoder_num_layer = 0
	decoder_num_unit = 256
	decoder_ratio_dropout = 0.0
	decoder_dim_output = output_dim

	# 途中までおこなった学習済みデータを使って学習を進める
	use_middle_trained_model = True

	# 何EPOCHずつモデルを保存するか
	epoch_step_save_model = 10

	# テストのパラメータ
	test_start_epoch = 1
	test_end_epoch = 50
	test_batch_size = 512

	# テスト時の出力値を保存するか
	test_probability_output = False
	test_probability_output_epoch = 50

	# ラべルの重みを付けるか
	label_balanced = False

	data_dir = '/n/work1/inoue/socialsignal/'
	src_dir = '/home/inoue/sapwork/socialsignal/'

	task = 'bc'

	check_label = 'no'

	# 出力ファイル名はパラメータに応じて動的に変更
	# 継承したクラスでクラス変数でパラメータを変更した後に，__init__でこの関数を呼び出すとよい
	def make_filenames(self):

		self.param_id = '%s_%s-bs%d-enc%d_%d-dr%02d-lr%s-wd%s' % (
			self.model_type,
			self.task,
			self.batch_size,
			self.encoder_num_unit,
			self.encoder_num_layer,
			self.encoder_ratio_dropout * 10,
			self.learning_rate,
			self.weight_decay
		)

		if self.do_z_normalize:
			self.param_id += '-znorm'

			if self.type_z_normalize != 'each':
				self.param_id += '-' + self.type_z_normalize
		
		if self.spec_augmentation:
			self.param_id += '-specaug'
		
		if self.do_oversampling:
			self.param_id += '-ovs'
		
		if self.label_balanced:
			self.param_id += '-balanced'
		
		if self.loss_function == 'focal':
			self.param_id += '-focalloss-gamma%d' % (self.focal_loss_gamma * 10)
		
		# あとでこちらに変更
		# if self.model_type in ['seq2one_cnn_lstm', 'seq2one_cnn_gru']:
		# 	self.param_id += '-cnn%d' % (
		# 		self.cnn_num_filters
		# 	)
		
		if self.model_type == 'seq2one_cnn_lstm':
			self.param_id += '-cnn%d' % (
				self.cnn_num_filters
			)
		
		if self.encoder_is_bidirectional == False:
			self.param_id += '-unidi'

		# 学習済みモデルの出力ディレクトリ名
		self.dir_model_output = self.data_dir + 'model/' + self.param_id

		# 学習ログの出力ファイル名
		self.filename_log_output = self.src_dir + 'log_train/' + self.param_id

		# テストログの出力ファイル名
		self.filename_log_test_output = self.src_dir + 'log_test/' + self.param_id

		# テスト結果ファイル名
		self.filename_result_output = self.src_dir + 'result/' + self.param_id

		# テスト結果ファイルをまとめたもの（交差検定向け）
		self.filename_result_summarize_output = self.src_dir + 'result_summarize/' + self.param_id

		# テストにおいて出力値を保存
		self.filename_output_test_output = self.src_dir + 'output_test/' + self.param_id
