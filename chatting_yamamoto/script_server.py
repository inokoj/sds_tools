import socket
import traceback
import threading
import time
import os
import xml.etree.ElementTree as ET

#print_lock = threading.Lock()
class Server:
	def __init__(self, host, port, parent_script):
		# Server Address (port)
		self.HOST = host
		self.PORT = port
		self.soc = None
		self.parent = parent_script

	def start_connecting(self):
		# Establishing a server
		self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.soc.bind((self.HOST, self.PORT))
		self.soc.listen()
		print('Waiting for connection...')

		try:
			while True:
				self.client_soc, client_addr = self.soc.accept() # Keep running this loop
				print('Connected at ', client_addr)

				left_msg = ''
				while True:
					
					if len(left_msg) == 0:
						message = self.client_soc.recv(1024).decode('UTF-8')
					else:
						message = left_msg
					
					if len(message.splitlines()) > 1:
						temp = message.splitlines()
						message = temp[0]
						left_msg = os.linesep.join(temp[1:])
					else:
						left_msg = ''
					
					root = ET.fromstring(message)
					label = root.tag

					if label == 'DIALOGUE_HUMAN':
						utterance = root[0].text
						id = root[1].text

						response = self.parent.bot.recieve_asr(utterance)
						self.send_utterance(response, id)
					
					if label == 'DIALOGUE_ROBOT':
						utterance = root[0].text
						id = root[1].text

						self.parent.bot.done_system_utterance(utterance)
					
					if label == 'TURN':
						state = root[0].text
						self.parent.bot.changed_turn_taking_state(state)

		except Exception as e:
			print(e)
			print('Disconnected, waiting...')
	
	def send_message(self, message):
		try:
			self.client_soc.send((message + "\n").encode('utf-8'))
		except Exception as e:
			print(e)
			print("Client disconnected!")
	
	def send_utterance(self, utterance, asr_id):
		response = '<RESPONSE><UTTERANCE>%s</UTTERANCE><ID>%s</ID></RESPONSE>' % (utterance, asr_id)
		self.send_message(response)