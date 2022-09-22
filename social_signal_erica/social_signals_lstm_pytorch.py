# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import socket
import struct
import copy, datetime
import time
import importlib

import torch
import torch.nn as nn
import torch.optim as optim

#from sklearn.preprocessing import StandardScaler
import argparse
import shutil
import math
import threading
from tkinter import Tk, Frame, BOTH, Label, Canvas
import subprocess

import configparser
import argparse
import json
import joblib

import pickle

# Original
import mytorch_util as util
from model_seq2one import seq2one_LSTM
from model_seq2one_gru import seq2one_GRU

USE_GPU = True
GPU_ID = 0

device = torch.device("cuda:%d" % GPU_ID if torch.cuda.is_available() else "cpu")

#INPUT_DIM = 40
#DROPOUT_RATE = 0.0
#NUM_HIDDEN_LAYERS = 5
#NUM_HIDDEN_UNITS = 256
#NUM_OUTPUT_DIM = 3	# 相槌，相槌以外，blank

#DO_Z_NORMALIZE = True
#DO_NORMALIZE_STATIC = True	# FBANKの最初の40次元（スタティックのみ）をZ正規化

#
# Adin tools setting
#

ADIN_PATH = ".\\adintool_windows\\adintool.exe"
ADIN_INPORT = 5900
ADIN_SERVER = "127.0.0.1"
ADIN_PARAMTYPE = "FBANK_D_A"
ADIN_VECLEN = 120
ADIN_HTKCONF = ".\\adintool_windows\\config.lmfb"
ADIN_PORT = 6632

THRESHOLD_LAUGH_DETECT = 0.5
THRESHOLD_LAUGH_SHARE = 0.7
THRESHOLD_LAUGH_TYPE = 0.5

#class for creating server so ISHIKI can connect to it
class ISHIKI_Server():

	def __init__(self, parent):
		self.client_socket = None
		self.is_connected = False
		self.parent = parent

	def connect(self, ishikiserver_port):
		# global ishikiserversock, ishikiserver_host, ishikiserver_port, ishikiclientsock, ishikiclient_address
		ishikiserversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		#ishikiserversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		ishikiserversock.bind(('', ishikiserver_port))
		ishikiserversock.listen(10)
		while True:
			output_log('Waiting for connection from I.S.H.I.K.I ...')
			output_log('at port %d' % ishikiserver_port)
			self.client_socket, ishikiclient_address = ishikiserversock.accept()
			output_log('Connected from I.S.H.I.K.I. from')
			output_log(ishikiclient_address)
			self.is_connected = True
			self.parent.notify_ishiki_connection(True)

			while self.is_connected:
				time.sleep(1)
			print("Reconnecting...")

	def send_message(self, message):
		print(message)
		try:
			self.client_socket.send(message.encode())
		except:
			print("ISHIKI disconnected!")
			self.is_connected = False
			self.parent.notify_ishiki_connection(False)

#class for creating client which connects to ISHIKI (for multiparty scenarios)
class ISHIKI_Multiparty_Client():

	def __init__(self, parent):
		self.server_socket = None
		self.is_connected = False
		self.parent = parent

	def connect(self, ishikiserver_host, ishikiserver_port):
		# global ishikiserversock, ishikiserver_host, ishikiserver_port, ishikiclientsock, ishikiclient_address
		self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		#ishikiserversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		try:
			self.server_socket.settimeout(10)
			print("sdfdsf")
			self.server_socket.connect((ishikiserver_host, ishikiserver_port))
			self.is_connected = True
			self.parent.notify_ishiki_connection(True)
			print("Connected to I.S.H.I.K.I.!")
			while self.is_connected:
				time.sleep(1)
			print("Disconnected from ISHIKI - trying to reconnect...")
		except Exception as e:
			print(e)

		self.connect(ishikiserver_host, ishikiserver_port)

	def send_message(self, message):
		print(message)
		try:
			self.server_socket.send(message.encode())
		except:
			print("ISHIKI disconnected!")
			self.is_connected = False
			self.parent.notify_ishiki_connection(False)

# ログを出力する（標準出力も行う）
#
def output_log(sentence):
	#global log_filename
	print(sentence)
	# with open('./models/%s/log.txt' % run_name, file_open_type) as f:
	# 	print(sentence, file=f)

#
# Adintoolsと接続
#
def connection_adin(adinserver_port):
	# global adinserversock, adinserver_host, adinserver_port, adinclientsock, adinclient_address
	adinserversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	#adinserversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

	adinserversock.bind(('', adinserver_port))
	adinserversock.listen(1)

	output_log('Waiting for connection from adintool ...')
	output_log('at port %d' % adinserver_port)
	adinclientsock, adinclient_address = adinserversock.accept()
	output_log('Connected from Adintool from')
	output_log(adinclient_address)
	return adinclientsock

def run_shared_laughter_model(audio_sample, scaler, model):
	feature_vector = []
	for ind in range(40):
		feature_vector.append(np.mean([float(y[ind]) for y in audio_sample]))
		feature_vector.append(np.std([float(y[ind]) for y in audio_sample]))
	feature_vector = np.array(feature_vector)
	feature_vector = feature_vector.reshape(1, -1)
	feature_norm = scaler.transform(feature_vector)
	response_probabilities = model.predict_proba(feature_norm)
	response_prob = response_probabilities[0][1]
	print(response_prob)
	return response_prob

# class Model(Chain):
# 	def __init__(self):
# 		initializer = initializers.Uniform(0.1)
# 		super(Model, self).__init__()
# 		with self.init_scope():
# 			self.bi_lstm = L.NStepBiLSTM(n_layers=5, in_size=INPUT_DIM * FRAME_STACKING, out_size=NUM_HIDDEN_UNITS, dropout=DROPOUT_RATE)
# 			self.linear = L.Linear(NUM_HIDDEN_UNITS * 2, NUM_OUTPUT_DIM, initialW=initializer)
	
# 	def __call__(self, x):
# 		hx = None
# 		cx = None
# 		hy, cy, ys = self.bi_lstm(hx=hx, cx=cx, xs=x)
# 		hs = [self.linear(F.dropout(h_tmp, ratio=DROPOUT_RATE)) for h_tmp in ys]
# 		#
# 		# CTCはsoft-maxは不要
# 		#
# 		return hs

class SocialSignalPrediction():
	def __init__(self, filename_config):

		config = configparser.ConfigParser()
		config.read(filename_config, encoding='utf-8')

		# 設定ファイル読み込み
		self.adinserver_host = config['ADIN'].get('adinserver_host')
		self.adinserver_port = config['ADIN'].getint('adinserver_port')
		self.use_multiparty_server = config['ISHIKI'].getboolean('multiparty')
		self.ishikiserver_host = config['ISHIKI'].get('ishikiserver_host')
		self.ishikiserver_port = config['ISHIKI'].getint('ishikiserver_port')
		self.target_human_id = config['ISHIKI'].getint('target_human_id')

		# 設定ファイル読み込み（shared laughter）
		self.do_shared_laughter = config['SHARED_LAUGH'].getboolean('do_shared_laughter')
		if self.do_shared_laughter:
			#self.filename_shared_laugh_model_scaler = config['SHARED_LAUGH'].get('filename_shared_laugh_model_scaler')
			self.filename_shared_laugh_model = config['SHARED_LAUGH'].get('filename_model_sl')
			self.filename_shared_laugh_type_model = config['LAUGH_TYPE'].get('filename_model_lt')

		# 設定ファイル読み込み（モデル）
		self.MODEL_DEFS = {}
		for model_type in eval(config['MODELS'].get('types')):
			name = config[model_type].get('name')
			filename_model_def = config[model_type].get('filename_model_def').strip()
			filename = config[model_type].get('filename_model').strip()
			filename_zscore = config[model_type].get('filename_zscore')
			num_dim = config[model_type].getint('num_dim')
			target_index = config[model_type].getint('target_index')
			do_process = config[model_type].getboolean('do_process')
			self.MODEL_DEFS[name] = [filename, num_dim, target_index, filename_model_def, filename_zscore, do_process]

		self.ishiki_server = None
		self.adinclientsock = None
		self.GUI = None

		t1 = threading.Thread(target=self.runModels, args=())
		t1.setDaemon(True)
		t1.start()

	def notify_ishiki_connection(self, isconnected):
		if isconnected:
			self.GUI.ishikiconnected()
		else:
			self.GUI.ishikidisconnected()

	#
	# モデルの読み込み
	#
	def runModels(self):

		#loss_func = nn.BCEWithLogitsLoss()

		torch.backends.cudnn.enabled = False

		MODELS = {}
		MODEL_PYTORCH_DEFS = {}
		ZSCORES = {}	# 学習データ全体での平均と分散
		DO_PROCESS = {}
		for type_ss, model_info in self.MODEL_DEFS.items():

			output_log("----------------------")
			output_log("  load model (type = %s)" % type_ss)
			output_log("----------------------")

			model_filename = model_info[0]
			print(model_filename)

			model_def_filename = model_info[3]
			print(model_def_filename)
			
			sys.path.append(os.path.dirname(model_def_filename))
			model_def = importlib.import_module(os.path.splitext(os.path.basename(model_def_filename))[0])
			md = model_def.ModelDef()

			if os.path.isfile(model_filename) == False:
				output_log('[ERROR] model file was not found : %s' % model_filename)
				sys.exit()
			
			if md.model_type == 'seq2one':
				model = seq2one_LSTM(md, isgpu=USE_GPU, gpuid=GPU_ID)
			
			if md.model_type == 'seq2one_gru':
				model = seq2one_GRU(md, isgpu=USE_GPU, gpuid=GPU_ID)
			
			# with open(model_filename, 'rb') as f:
			# 	model = cloudpickle.load(f)
			# 	output_log('Loaded : ' + model_filename)

			param = torch.load(model_filename)
			model.load_state_dict(param)
			print('Loaded (model): ' + model_filename)

			if USE_GPU == True:
				print('Transferring the model to GPU (ID=%d) ....' % GPU_ID)
				model.cuda(GPU_ID)
		
			model.eval()
			output_log('Loaded : ' + model_filename)

			if USE_GPU == True:
				model.to(device)
			
			# 学習データ全体で平均と分散を計算する
			if md.do_z_normalize and md.type_z_normalize == 'whole':
				print('##### Load whole mean and var values #####')
				print(model_info[4])
				zscore = np.loadtxt(model_info[4])
				ZSCORES[type_ss] = zscore
			
			MODELS[type_ss] = model
			MODEL_PYTORCH_DEFS[type_ss] = md
			DO_PROCESS[type_ss] = model_info[5]

		if self.do_shared_laughter:
			
			print("Loading shared laugh model...")
			if os.path.isfile(self.filename_shared_laugh_model) == False:
				output_log('[ERROR] model file was not found : %s' % self.filename_shared_laugh_model)
				sys.exit()

			with open(self.filename_shared_laugh_model, 'rb') as f:
				sl_model = pickle.load(f)
				sl_sc = pickle.load(f)

			# self.shared_laugh_model_scaler = joblib.load(self.filename_shared_laugh_model_scaler)
			# self.shared_laugh_model = joblib.load(self.filename_shared_laugh_model)

			print("Loading shared laugh type model...")
			if os.path.isfile(self.filename_shared_laugh_type_model) == False:
				output_log('[ERROR] model file was not found : %s' % self.filename_shared_laugh_type_model)
				sys.exit()

			with open(self.filename_shared_laugh_type_model, 'rb') as f:
				lt_model = pickle.load(f)
				lt_sc = pickle.load(f)

		self.GUI.modelLoadingFinished()

		#runs adintool
		# subprocess.Popen("%s -in adinnet -inport %d -out vecnet -server %s -paramtype %s -veclen %d -htkconf %s -port %d" % \
		# 	(ADIN_PATH, ADIN_INPORT, ADIN_SERVER, ADIN_PARAMTYPE, ADIN_VECLEN, ADIN_HTKCONF, ADIN_PORT), shell=True)
		
		self.adinclientsock = connection_adin(self.adinserver_port)

		# make the ishiki connection in a new thread - we can still do predictions even without connecting to it
		#if multiparty connection make a client to connect to ISHIKI's server
		if self.use_multiparty_server:
			self.ishiki_server = ISHIKI_Multiparty_Client(self)
			t = threading.Thread(target=self.ishiki_server.connect, args=(self.ishikiserver_host, self.ishikiserver_port,))
			t.daemon = True
			t.start()
		#else, make a server so ISHIKI can connect to it 
		else:
			self.ishiki_server = ISHIKI_Server(self)
			t = threading.Thread(target=self.ishiki_server.connect, args=(self.ishikiserver_port,))
			t.daemon = True
			t.start()


		input_features = []

		sigmoid = nn.Sigmoid()

		while True:

			#try:
				
			self.adinclientsock.send(b'a')

			rcvmsg = self.adinclientsock.recv(4)
			nbytes = int.from_bytes(rcvmsg, 'big')
			#nbytes = struct.unpack('=i', rcvmsg)[0]

			#print('start rec')

			rcvmsg = b''
			while True:
				rcvmsg_part = self.adinclientsock.recv(1000)
				#print(rcvmsg_part)
				rcvmsg += rcvmsg_part
				if len(rcvmsg) >= nbytes:
					break
			#print(rcvmsg)
			#rcvmsg = rcvmsg[:-1]

			if len(rcvmsg) == 0:
				continue
			
			recieved_features = np.fromstring(rcvmsg, dtype=np.float32)

			input_features = np.reshape(recieved_features, (-1, 40))
			
			#for the first prediction, turn on the GUI because we know it is now connected
			if self.GUI.isLoaded == False:
				self.GUI.adinconnected()

			# ここで発話終了
			data_x = [np.array(input_features)]
			
			feat_mat = input_features
			feat_mat = np.array(feat_mat)

			# # Z正規化
			# if DO_Z_NORMALIZE:
			# 	xmean = np.mean(feat_mat, axis=0)
			# 	xstd  = np.std(feat_mat, axis=0)
			# 	feat_mat_znorm = (feat_mat - xmean) / xstd

			# feat_mat = np.array(feat_mat)
			input_len = np.array(input_features).shape[0]
			len_x = [input_len]
			

			# タイムスタンプ
			ts = int(datetime.datetime.now().timestamp() * 1000)	# ミリ秒に合わせる
			output_log("Time : %d" % ts)

			#print(feat_mat.shape)
			self.GUI.updateSignalStats((input_len + 6) * 10, ts)

			#data_x = [torch.tensor(feat_mat, device=device, dtype=torch.float32)]
			#data_x_znorm = [torch.tensor(feat_mat_znorm, device=device, dtype=torch.float32)]
			#len_x = [input_len]
			
			#print(feat_mat)
			for type_ss, model in MODELS.items():
				
				do_process = DO_PROCESS[type_ss]
				
				probability = 0.
				if do_process:
					md = MODEL_PYTORCH_DEFS[type_ss]
					data_x_temp = data_x
					len_x_temp = len_x

					# フレームスタッキング
					if md.frame_stack > 1:
						data_x_temp, len_x_temp = util.frame_stack(data_x_temp, len_x_temp, md.frame_stack)
				
					# Zスコア正規化
					if md.do_z_normalize and md.type_z_normalize == 'each':
						data_x_temp = util.z_normalize(data_x_temp)
				
					# 学習データセット全体の平均と分散で正規化
					if md.do_z_normalize and md.type_z_normalize == 'whole':
						mean_whole = ZSCORES[type_ss][0]
						var_whole = ZSCORES[type_ss][1]
						data_x_temp = util.z_normalize_param(data_x_temp, mean_whole, var_whole)
				
					# Variableを作成
					data_x_ = [torch.tensor(data, device=device, dtype=torch.float32) for data in data_x_temp]
					len_x_ = torch.tensor(len_x_temp, dtype=torch.int64)

					#num_output_dim = MODEL_DEFS[type_ss][2]
					target_dim_index = self.MODEL_DEFS[type_ss][2]

					outputs = model(data_x_, len_x_)
					#outputs = model(data_x_znorm, len_x)

					# print(outputs.data)
					# print(sigmoid(outputs.data))

					# Soft-max
					sigmoid_sum = torch.sum(sigmoid(outputs))
					sigmoid_target = sigmoid(outputs).data[0][target_dim_index]
					
					probability = float((sigmoid_target / sigmoid_sum).data.item())

				output_log('p(%s)=%.4f' % (type_ss, probability))
				msg_result = "%s,%d,%f,%d\n" % (type_ss, ts, probability, self.target_human_id)

				#update GUI
				if type_ss == 'backchannel':
					self.GUI.updateBC(probability)
				elif type_ss == 'laugh':
					self.GUI.updateLaugh(probability)

				if self.ishiki_server.is_connected:
					self.ishiki_server.send_message(msg_result)
				
				#
				# For shared laugh
				#
				if type_ss == 'laugh' and self.do_shared_laughter:
					
					

					#
					# 共有笑い
					#
					input_features_temp = []
					for idx_feat in range(len(input_features)):
						if input_features[idx_feat, 0] > 3:
							input_features_temp.append(input_features[idx_feat])
						
					input_features = np.array(input_features_temp)

					f_fbank_mean = np.mean(input_features, 0)
					f_fbank_std = np.std(input_features, 0)
					# f_fbank_max = np.max(input_features, 0)
					# f_fbank_min = np.min(input_features, 0)
					# f_fbank_range = f_fbank_max - f_fbank_min

					f_fbank_concat = [np.concatenate([f_fbank_mean, f_fbank_std])]
					data_x = sl_sc.transform(f_fbank_concat)
					sl_probability = sl_model.predict_proba(data_x)[0][1]
					p = sl_model.predict_proba(data_x)
					output_log('p(shared)=%.4f' % sl_probability)

					self.GUI.updateSharedLaugh(sl_probability)

					#
					# 笑いの種類の選択
					#

					data_x = lt_sc.transform(f_fbank_concat)
					lt_probability = lt_model.predict_proba(data_x)[0][1]
					output_log('p(type)=%.4f' % lt_probability)
					
					#shared_laugh_prob, laught_type = run_shared_laughter_model(shared_feat_mat, self.shared_laugh_model_scaler, self.shared_laugh_model)
					#output_log('p(%s)=%.4f' % ("shared laugh", shared_laugh_prob))
					msg_result = "%s,%d,%f,%f,%f,%d\n" % ("shared laugh", ts, probability, sl_probability, lt_probability, self.target_human_id)
					
					self.GUI.updateSharedLaugh(sl_probability)
					self.GUI.updateLaughType(lt_probability)

					if self.ishiki_server.is_connected:
						self.ishiki_server.send_message(msg_result)

			print("")
			input_features = []
			# except:
			# 	self.adinclientsock.close()
			# 	output_log('Adintool was disconnected!!')
			# 	connection_adin(self.adinserver_port)
			# 	continue

		self.adinclientsock.close()
		# ishikiclientsock.close()

class GUI(Frame):
	def __init__(self, parent):
		Frame.__init__(self, parent)
		self.parent = parent
		self.parent = parent
		self.social_signal_analyzer = ssa
		self.isLoaded = False
		ssa.GUI = self
		self.initUI()

	def initUI(self):
		self.parent.title("Social Signal Analyzer")
		self.pack(fill=BOTH, expand=1)

		self.adinconstatus = Label(text = "Waiting for models to load...", fg = "Red")
		self.adinconstatus.place(x = 30, y = 10)

	def modelLoadingFinished(self):
		self.adinconstatus.config(text='Check connection by speaking')
		self.afterLoad()

	def afterLoad(self):

		self.ishikiconstatus = Label(text = "ISHIKI not connected", fg = "Red")
		self.ishikiconstatus.place(x = 30, y = 30)

		self.human_id_label = Label(text="Human ID = " + str(self.social_signal_analyzer.target_human_id), fg = "Black")
		self.human_id_label.place(x = 30, y = 50)

		self.canvas = Canvas(self)

		self.bg = self.canvas.create_rectangle(120, 70, 550, 201, fill = "White")
		self.bc_rec = self.canvas.create_rectangle(0,0,0,0,fill="yellow")
		self.laugh_rec = self.canvas.create_rectangle(0,0,0,0,fill="orange red")
		self.shared_laugh_rec = self.canvas.create_rectangle(0,0,0,0,fill="dodger blue")
		self.laugh_type_rec = self.canvas.create_rectangle(0,0,0,0,fill="green")
		self.canvas.pack(fill=BOTH, expand=1)

		self.bc_label = Label(text = "P (Backchannel)")
		self.bc_label.place(x=120, y= 210)

		self.laugh_label = Label(text = "P (Laugh)")
		self.laugh_label.place(x=240, y= 210)

		self.shared_laugh_label = Label(text = "P (Shared)")
		self.shared_laugh_label.place(x=360, y= 210)

		self.laugh_type_label = Label(text = "P (Type)")
		self.laugh_type_label.place(x=480, y= 210)

		self.bc_percent = Label(text = "", font = "-weight bold", bg = "White")
		self.laugh_percent = Label(text = "", font = "-weight bold", bg = "White")
		self.shared_laugh_percent = Label(text = "", font = "-weight bold", bg = "White")
		self.laugh_type_percent = Label(text = "", font = "-weight bold", bg = "White")

		self.sig_time = Label(text = "", font = "-weight bold")
		self.sig_time.place(x = 10, y = 100)
		self.sig_length = Label(text = "", font = "-weight bold")
		self.sig_length.place(x = 10, y = 130)


	def updateBC(self, prob):
		self.canvas.coords(self.bc_rec,140, 200 - (prob*100), 170 ,200)
		self.bc_percent.place(x = 140, y = 75)
		self.bc_percent.config(text= str(int(prob*100)) + "%")

	def updateLaugh(self, prob):
		self.canvas.coords(self.laugh_rec, 250, 200 - (prob*100), 280 ,200)
		self.laugh_percent.place(x = 250, y = 75)
		self.laugh_percent.config(text= str(int(prob*100)) + "%")
	
	def updateSharedLaugh(self, prob):
		self.canvas.coords(self.shared_laugh_rec, 360, 200 - (prob*100), 390 ,200)
		self.shared_laugh_percent.place(x = 360, y = 75)
		self.shared_laugh_percent.config(text= str(int(prob*100)) + "%")
	
	def updateLaughType(self, prob):
		self.canvas.coords(self.laugh_type_rec, 470, 200 - (prob*100), 500 ,200)
		self.laugh_type_percent.place(x = 470, y = 75)
		self.laugh_type_percent.config(text= str(int(prob*100)) + "%")


	def updateSignalStats(self, length, ts):
		self.sig_length.config(text = str(length) + " ms")
		a = str(datetime.datetime.fromtimestamp(ts/1e3).time())
		self.sig_time.config(text = a.split('.')[0])

	def adinconnected(self):
		self.adinconstatus.config(text="ADIN connected", fg = "Green")

	def ishikiconnected(self):
		self.ishikiconstatus.config(text="ISHIKI connected", fg = "Green")

	def ishikidisconnected(self):
		self.ishikiconstatus.config(text="ISHIKI not connected", fg = "Red")


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Social signal classifier')
	parser.add_argument('config', help='Configration file path')
	args = parser.parse_args()

	ssa = SocialSignalPrediction(args.config)
	root = Tk()
	root.geometry("600x250+300+300")
	GUI(root)
	root.mainloop()