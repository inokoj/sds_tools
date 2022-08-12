import os, sys

class Bot:

	def __init__(self):
		pass
	
	# ユーザ発話が認識する度に呼び出される
	# 応答文を返す
	def recieve_asr(self, utterance):
		print('Recieved User Utterance: ' + utterance)

		response = 'ここに返事を入力する'

		return response

	# システムが発話を行ったら呼び出される
	# 生成した発話が使用されたことがここで確定できる
	def done_system_utterance(self, utterance):
		print('System Uttered: ' + utterance)
	
	# ターンテイキングの状態が変更になったら呼び出される
	def changed_turn_taking_state(self, state):
		print('Changed turn-taking state: ' + state)


	