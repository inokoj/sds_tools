# -*- coding: utf-8 -*-

# Gen minibatchとかを書く
# そこでデータを読み込む？

import os, sys
import numpy as np
import torch
from collections import defaultdict
import copy
import random

# ログファイルに出力しつつコンソールにも表示
def out_log(sentence, filename=None):

	print(sentence)

	if filename is not None:
		with open(filename, 'a') as f:
			f.write(('%s' % sentence) + '\n')

# データセットのラベルの分布を返す
def calc_label_dist(list_dataset, name_label):

	# データ自体を読み込む
	dist_label = defaultdict(int)

	# 指定された種類のデータとラベルを読み込む
	for data_each in list_dataset:
		
		data = np.load(data_each, allow_pickle=True)
		l = data['label'].tolist()[name_label]
		dist_label[l] += 1
	
	return dist_label

# データセットのラベルの分布を返す
def calc_label_dist_multiclass(list_dataset, list_name_label, type_multiclass):

	# データ自体を読み込む
	dist_label = defaultdict(int)

	# 指定された種類のデータとラベルを読み込む
	for data_each in list_dataset:
		
		data = np.load(data_each, allow_pickle=True)
		
		l_found = 0
		for idx, name_label in enumerate(list_name_label):
			l = data['label'].tolist()[name_label]
			if l == 1:
				if type_multiclass == 'other':
					l_found = idx + 1
				else:
					l_found = idx
		
		dist_label[l_found] += 1

	return dist_label

# データセット全体の平均と分散を計算する
def calc_zscore(list_dataset, name_data, dim_feat, num_stack=1):

	# まずはサンプル数を計算
	num_frame_total = 0
	for data_each in list_dataset:
		data = np.load(data_each, allow_pickle=True)
		d = data['data'].tolist()[name_data]
		
		if num_stack > 1:
			x, l = frame_stack([d], [len(d)], num_stack)
			d = x[0]
		
		num_frame_total += len(d)

	# 続いて平均と二乗平均を計算
	mean = np.zeros(dim_feat)
	mean_sq = np.zeros(dim_feat)
	for data_each in list_dataset:
		data = np.load(data_each, allow_pickle=True)
		d = data['data'].tolist()[name_data]

		if num_stack > 1:
			x, l = frame_stack([d], [len(d)], num_stack)
			d = x[0]
		
		mean += np.sum(d, axis=0)
		d = np.array(d)
		mean_sq += np.sum(d*d, axis=0)

	mean /= num_frame_total
	mean_sq /= num_frame_total

	# 分散も求める E[x^2] - E[x]^2
	var = mean_sq - (mean*mean)

	return mean, var

# 指定されたラベルが1がどうかをチェック
def check_label(filename, name_label):
	
	data = np.load(filename, allow_pickle=True)
	
	if data['label'].tolist()[name_label] == 1:
		result = True
	else:
		result = False

	return result

# 分割されたバッチデータセットを読み込む
def load_batch_data(list_data_minibatch, name_data, name_label):
	
	# データ自体を読み込む
	data_x = []
	data_y = []
	len_x = []
	len_y = []

	data_temp = []
	data_meta = []

	# 指定された種類のデータとラベルを読み込む
	for data_minibatch in list_data_minibatch:
		
		data = np.load(data_minibatch, allow_pickle=True)
		
		d = data['data'].tolist()[name_data]
		l = data['label'].tolist()[name_label]

		# メタデータ
		session_id = data['data'].tolist()['session_id']
		start_time = data['data'].tolist()['start_time']
		end_time = data['data'].tolist()['end_time']
		speaker = data['data'].tolist()['speaker']

		# ラベルの系列長が１のときは２にする
		if isinstance(l, int):
			_l = np.zeros(2)
			_l[l] = 1.
			l = _l
		
		# ラベルの系列長だけ場合分けする
		if type(l) is list:
			len_l = len(l)
		else:
			len_l = 1

		data_temp.append([d, len(d), l, len_l, session_id, speaker, start_time, end_time, ])
	
	# 系列長で降順にソート
	data_temp = sorted(data_temp, key=lambda x:x[1])
	data_temp.reverse()

	for d in data_temp:
		data_x.append(d[0])
		len_x.append(d[1])
		data_y.append(d[2])
		len_y.append(d[3])

		data_meta.append([d[4], d[5], d[6], d[7]])
	
	data_x = np.array(data_x)
	data_y = np.array(data_y)
	len_x = np.array(len_x)
	len_y = np.array(len_y)

	return data_x, data_y, len_x, len_y, data_meta

# 分割されたバッチデータセットを読み込む
# ラベルが複数種類ある場合
def load_batch_data_multiclass(list_data_minibatch, name_data, list_name_label, type_multiclass):
	
	# データ自体を読み込む
	data_x = []
	data_y = []
	len_x = []
	len_y = []

	data_temp = []
	data_meta = []

	# type_multiclass
	# "other" ... 各ラベルがlist_name_labelで指定されており，これらのいずれにもあてはまらない場合の次元を作る

	# 指定された種類のデータとラベルを読み込む
	for data_minibatch in list_data_minibatch:
		
		data = np.load(data_minibatch, allow_pickle=True)
		
		d = data['data'].tolist()[name_data]
		
		# 笑いなしを１次元目に
		l = 0
		for idx, name_label in enumerate(list_name_label):
			l = data['label'].tolist()[name_label]
			if l == 1:
				if type_multiclass == 'other':
					l = idx + 1
				else:
					l = idx

		# メタデータ
		session_id = data['data'].tolist()['session_id']
		start_time = data['data'].tolist()['start_time']
		end_time = data['data'].tolist()['end_time']
		speaker = data['data'].tolist()['speaker']

		len_l = 1

		data_temp.append([d, len(d), l, len_l, session_id, speaker, start_time, end_time])
	
	# 系列長で降順にソート
	data_temp = sorted(data_temp, key=lambda x:x[1])
	data_temp.reverse()

	for d in data_temp:
		data_x.append(d[0])
		len_x.append(d[1])
		data_y.append(d[2])
		len_y.append(d[3])

		data_meta.append([d[4], d[5], d[6], d[7]])
	
	data_x = np.array(data_x)
	data_y = np.array(data_y)
	len_x = np.array(len_x)
	len_y = np.array(len_y)

	return data_x, data_y, len_x, len_y, data_meta

# # 分割されたバッチデータセットのメタデータを読み込む
# def load_batch_meta_data(list_data_minibatch, name_data):

# 	data_extracted = []

# 	# 指定された種類のデータとラベルを読み込む
# 	for data_minibatch in list_data_minibatch:
		
# 		data = np.load(data_minibatch, allow_pickle=True)
		
# 		d = data['data'].tolist()[name_data]

# 		session_id = data['data'].tolist()['session_id']
# 		start_time = data['data'].tolist()['start_time']
# 		end_time = data['data'].tolist()['end_time']
# 		speaker = data['data'].tolist()['speaker']

# 		data_extracted.append([session_id, speaker, start_time, end_time])
	
# 	# 系列長で降順にソート
# 	data_temp = sorted(data_temp, key=lambda x:x[1])
# 	data_temp.reverse()
	
# 	return data_extracted

# 分割されたバッチデータセットを読み込む
# 特徴量が2種類の場合
def load_batch_data_multimodal(list_data_minibatch, name_data1, name_data2, name_label):
	
	# データ自体を読み込む
	data_x1 = []
	data_x2 = []
	data_y = []
	len_x1 = []
	len_x2 = []
	len_y = []

	data_temp = []

	# 指定された種類のデータとラベルを読み込む
	for data_minibatch in list_data_minibatch:
		
		data = np.load(data_minibatch, allow_pickle=True)
		
		d1 = data['data'].tolist()[name_data1]
		d2 = data['data'].tolist()[name_data2]
		l = data['label'].tolist()[name_label]

		# ラベルの系列長が１のときは２にする
		if isinstance(l, int):
			_l = np.zeros(2)
			_l[l] = 1.
			l = _l
		
		# ラベルの系列長だけ場合分けする
		if type(l) is list:
			len_l = len(l)
		else:
			len_l = 1

		data_temp.append([d1, len(d1), d2, len(d2), l, len_l])
	
	data_temp = sorted(data_temp, key=lambda x:x[1])
	data_temp.reverse()

	for d in data_temp:
		data_x1.append(d[0])
		len_x1.append(d[1])
		data_x2.append(d[2])
		len_x2.append(d[3])
		data_y.append(d[4])
		len_y.append(d[5])
	
	data_x1 = np.array(data_x1)
	data_x2 = np.array(data_x2)
	data_y = np.array(data_y)
	len_x1 = np.array(len_x1)
	len_x2 = np.array(len_x2)
	len_y = np.array(len_y)

	return data_x1, data_x2, data_y, len_x1, len_x2, len_y

# データをZ正規化（各次元で平均0, 分散1）
def z_normalize(list_data):
	
	list_data_new = []

	for data in list_data:
		xmean = np.mean(data, axis=0)
		xstd  = np.std(data, axis=0)
		
		# print('before')
		# print(xstd)
		xstd = np.array([1.0 if x == 0.0 else x for x in xstd])
		# print('after')
		# print(xstd)
		zscore = (data - xmean) / xstd
		list_data_new.append(zscore)
	
	return list_data_new

# データをZ正規化（平均と分散が与えられた場合）
def z_normalize_param(list_data, mean, var):
	
	std = np.sqrt(var)
	list_data_new = []
	for data in list_data:
		zscore = (data - mean) / std
		list_data_new.append(zscore)
	
	return list_data_new

# 二値分類の精度を測る
def calc_binary_classification_score(predict, target):

	sample = len(target)
	correct = predict.eq(target).sum().item()

	tp = torch.sum(predict * target).item()
	fp = torch.sum(predict * (1 - target)).item()
	fn = torch.sum((1 - predict) * target).item()
	tn = torch.sum((1 - predict) * (1 - target)).item()

	accuracy = correct / sample

	vals = {}
	#vals['accuracy'] = accuracy
	vals['tp'] = tp
	vals['fp'] = fp
	vals['fn'] = fn
	vals['tn'] = tn
	
	return vals

# 多クラス分類の精度を測る
def calc_multi_classification_score(predict, target, num_class):

	sample = len(target)
	#accuracy = correct / sample

	vals = {}
	vals['sample'] = sample

	correct = 0.

	for i in range(num_class):
		tp = 0.
		fp = 0.
		fn = 0.
		tn = 0.
		for j in range(sample):
			if predict[j] == i and target[j] == i:
				tp += 1
				correct += 1
			if predict[j] != i and target[j] == i:
				fn += 1
			if predict[j] == i and target[j] != i:
				fp += 1
			if predict[j] != i and target[j] != i:
				tn += 1
		vals['%d-tp'%i] = tp
		vals['%d-fn'%i] = fn
		vals['%d-fp'%i] = fp
		vals['%d-tn'%i] = tn
	
	vals['correct'] = correct
	
	return vals

# 二値分類の精度の結果をまとめる
def summary_binary_classification_score(scores):

	tp = scores['tp']
	fp = scores['fp']
	fn = scores['fn']
	tn = scores['tn']

	accuracy = float(tp + tn) / (tp + fp + fn + tn)
	
	if tp + fp > 0:
		one_precision = tp / (tp + fp)
	else:
		one_precision = 0.

	if tp + fn > 0:
		one_recall = tp / (tp + fn)
	else:
		one_recall = 0.

	if one_precision + one_recall > 0.:
		one_f1 = (2 * one_precision * one_recall) / (one_precision + one_recall)
	else:
		one_f1 = 0.
	
	if tn + fn > 0:
		zero_precision = tn / (tn + fn)
	else:
		zero_precision = 0.

	if tn + fp > 0:
		zero_recall = tn / (tn + fp)
	else:
		zero_recall = 0.

	if zero_precision + zero_recall > 0.:
		zero_f1 = (2 * zero_precision * zero_recall) / (zero_precision + zero_recall)
	else:
		zero_f1 = 0.

	macro_f1 = (one_f1 + zero_f1) / 2.

	vals = {}
	vals['accuracy'] = accuracy
	vals['one_precision'] = one_precision
	vals['one_recall'] = one_recall
	vals['one_f1'] = one_f1
	vals['zero_precision'] = zero_precision
	vals['zero_recall'] = zero_recall
	vals['zero_f1'] = zero_f1
	vals['macro_f1'] = macro_f1
	vals['tp'] = tp
	vals['fp'] = fp
	vals['fn'] = fn
	vals['tn'] = tn
	
	return vals

# 多クラス分類の精度の結果をまとめる
def summary_multi_classification_score(scores, num_class):

	vals = {}
	vals['accuracy'] = float(scores['correct']) / scores['sample']

	macro_f1 = 0.
	
	for idx in range(num_class):

		tp = scores['%d-tp'%idx]
		fp = scores['%d-fp'%idx]
		fn = scores['%d-fn'%idx]
		tn = scores['%d-tn'%idx]

		if tp + fp > 0:
			precision = tp / (tp + fp)
		else:
			precision = 0.
	
		if tp + fn > 0:
			recall = tp / (tp + fn)
		else:
			recall = 0.
	
		if precision + recall > 0.:
			f1 = (2 * precision * recall) / (precision + recall)
		else:
			f1 = 0.
		
		vals['%d-precision'%idx] = precision
		vals['%d-recall'%idx] = recall
		vals['%d-f1'%idx] = f1

		vals['%d-tp'%idx] = tn
		vals['%d-fp'%idx] = fp
		vals['%d-fn'%idx] = fn
		vals['%d-tn'%idx] = tn

		macro_f1 += f1
	
	macro_f1 /= num_class
	vals['macro_f1'] = macro_f1
	
	return vals

# フレームスタッキングする
def frame_stack(data_x, len_x, num_stack):

	# print("XXXXX")
	# print(data_x[0][:10])

	data_x_new = []
	len_x_new = []

	# サンプル
	for i in range(len(data_x)):
		num_frame = len(data_x[i])
		num_frame_stacked = int(num_frame / num_stack)
		dim_raw = len(data_x[i][0])
		data_x_i_new = np.zeros([num_frame_stacked, dim_raw * num_stack])

		# 新しいフレーム
		for j in range(num_frame_stacked):
			# 各スタック
			for k in range(num_stack):
				data_x_i_new[j, dim_raw*k : dim_raw*(k+1)] = data_x[i][j*num_stack+k]
		data_x_new.append(data_x_i_new)
		len_x_new.append(num_frame_stacked)
	
	data_x_new = np.array(data_x_new)
	len_x_new = np.array(len_x_new)

	#print(data_x_new[0][0:10])
	return data_x_new, len_x_new

# Spec augmentation
def spec_augment(x):
	
	aug_F = 15	# マスクする周波数ビンの最大幅
	aug_T = 20	# マスクする時間フレームの最大幅
	
	x = np.array(x)
	LMFB_DIM = x.shape[1]
	#print(x)
	
	aug_f = np.random.randint(0, aug_F)
	aug_f0 = np.random.randint(0, LMFB_DIM - aug_f)
	
	aug_f_mask_from = aug_f0
	aug_f_mask_to = aug_f0 + aug_f
	x[:, aug_f_mask_from:aug_f_mask_to] = 0.0
	
	if x.shape[0] > aug_T:
		aug_t = np.random.randint(0, aug_T)
		aug_t0 = np.random.randint(0, x.shape[0] - aug_t)
		aug_t_mask_from = aug_t0
		aug_t_mask_to = aug_t0 + aug_t
		x[aug_t_mask_from:aug_t_mask_to, :] = 0.0
	
	#print(x)

	return x

#
# オーバーサンプリング
# マジョリティのクラスにデータ数を合わせる
#
def oversampling(list_dataset, name_label):

	# 各クラスのデータ数を数え上げ
	num_data_by_label = defaultdict(int)
	for data in list_dataset:
		
		data = np.load(data, allow_pickle=True)
		l = data['label'].tolist()[name_label]

		num_data_by_label[l] += 1
	
	new_list_dataset = copy.deepcopy(list_dataset)

	# マジョリティクラスに合うようにオーバサンプリング
	max_num = max(num_data_by_label.values())
	for key in num_data_by_label.keys():
		
		# 対象となるクラスのデータを抜き出す
		list_target = []
		for filename in list_dataset:
		
			data = np.load(filename, allow_pickle=True)
			l = data['label'].tolist()[name_label]
			if l == key:
				list_target.append(copy.deepcopy(filename))

		# オーバーサンプリング
		num_append = num_data_by_label[key]
		while num_append < max_num:
			d = random.choice(list_target)
			new_list_dataset.append(d)
			num_append += 1
	
	random.shuffle(new_list_dataset)

	return new_list_dataset

