# -*- coding: utf-8 -*-

#
# 系列から長さ１のラベルを出力するモデル
#

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
#from mytorch import lstm_encoder
#from mytorch import mlp_decoder
import pdb

#
# LSTMを用いたモデル
#
class seq2one_LSTM(nn.Module):
	
	def __init__(self, model_def, isgpu, gpuid):
		
		super(seq2one_LSTM, self).__init__()
	
		self.inDim = model_def.input_dim			# 入力の次元

		# Encoder
		self.en_nlayer = model_def.encoder_num_layer			# Encoderのレイヤー数
		self.en_nhidden = model_def.encoder_num_unit			# Encoderのユニット数
		self.en_dr = model_def.encoder_ratio_dropout			# EncoderのDropout率
		self.bidirectFlag = model_def.encoder_is_bidirectional	# EncoderがBidrectionalか
		
		# Decoder
		self.de_nlayer = model_def.decoder_num_layer			# Decoderの層数
		self.de_nhidden = model_def.decoder_num_unit			# Decoderのユニット数
		self.de_dr = model_def.decoder_ratio_dropout			# DecoderのDropout率
		self.outDim = model_def.output_dim						# 出力の次元

		# 計算用
		self.multi = 2 if self.bidirectFlag else 1	# EncoderがBidrectionalの場合にはDecoderへ繋がるユニット数を２倍にする
		
		# GPU
		self.isgpu = isgpu				# GPUを使用するか
		self.gpuid = gpuid				# 使用するGPUのID
		
		# self.model_type = model_type
		# self.return_seq = return_seq # True for attention

		self.build()

	#
	# モデルに必要な変数を定義
	#
	def build(self):
		
		# Encoder
		self.encoder = nn.LSTM(
			input_size = self.inDim,
			hidden_size = self.en_nhidden,
			num_layers = self.en_nlayer,
			bidirectional = self.bidirectFlag,
			batch_first = True,
			dropout = self.en_dr
		)
	
		# Decoder
		# 隠れ層がある場合
		self.decoder = []
		if self.de_nlayer > 0:
			layer = [nn.Dropout(p=self.de_dr), nn.Linear(self.en_nhidden*self.multi, self.de_nhidden), nn.ReLU(), nn.Dropout(p=self.de_dr)] 
			self.decoder.extend(layer)
			for i in range(self.de_nlayer - 1):
				layer = [nn.Linear(self.de_nhidden, self.de_nhidden), nn.ReLU(), nn.Dropout(p=self.de_dr)] 
				self.decoder.extend(layer)
			# output layer
			layer = [nn.Linear(self.de_nhidden, self.outDim)] 
			self.decoder.extend(layer)

		# 隠れ層がなく出力層のみの場合
		else:
			# output layer
			#layer = [nn.Dropout(p=self.de_dr), nn.Linear(self.en_nhidden*self.multi, self.outDim)]
			layer = [nn.Linear(self.en_nhidden*self.multi, self.outDim)] 
			self.decoder.extend(layer)
		
		self.decoder = nn.ModuleList(self.decoder)
	
	#
	# モデルの計算フローを定義
	#
	def forward(self, x, x_len):
		
		# Encoder
		#print(x)
		#packed = nn.utils.rnn.pack_sequence(x, batch_first=True)
		# x, x_len = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
		
		padded = nn.utils.rnn.pad_sequence(x, batch_first=True)
		# print(padded)
		# print(x_len)
		packed = nn.utils.rnn.pack_padded_sequence(padded, x_len, batch_first=True, enforce_sorted=False)
		#print(packed)
		# h0, c0 = self.initialize(nbatch)
		ys, (h, c) = self.encoder(packed)
		
		# print(x)
		#print(ys)
		# print(ys[0])
		# print(len(ys[0]))
		#print(len(ys))
		# print(x_len)
		
		if self.bidirectFlag:
			ys, ys_len = nn.utils.rnn.pad_packed_sequence(ys, batch_first=True)
			#ys = nn.utils.rnn.pack_padded_sequence(ys, x_len, batch_first=True)
			#print(ys)
			decoder_h = torch.stack([ys[i, x_l - 1,:].float() for i, x_l in enumerate(x_len)])
		else:
			decoder_h = h[-1,:,:] # extract hidden value of last time
		
		#print(decoder_h)
		
		# decoder
		for layer in self.decoder:
			decoder_h = layer(decoder_h) 
		
		return decoder_h
	
	#
	# 初期状態を作成
	#
	def initialize(self, nbatch):
		
		if self.isgpu:
			h0 = Variable(
				torch.zeros(self.multi*self.en_nlayer, nbatch, self.en_nhidden).cuda(self.gpuid)
			).float()
		
			c0 = Variable(
				torch.zeros(self.multi*self.en_nlayer, nbatch, self.en_nhidden).cuda(self.gpuid)
			).float()
		else:
			h0 = Variable(
				torch.zeros(2*self.en_nlayer, nbatch, self.en_nhidden)
			).float()
		
			c0 = Variable(torch.zeros(2*self.en_nlayer, nbatch, self.en_nhidden)).float()
		
		return h0, c0
