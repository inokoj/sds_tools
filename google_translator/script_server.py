from os import linesep
import os
import socket
import traceback
import threading
import time

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
		# self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.soc.bind(('', self.PORT))
		self.soc.listen(10)
		print('Waiting for connection on ...', self.PORT)

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
					
					translated = self.parent.translate(message)
					self.send_message(translated)
					
		except Exception as e:
			print(e)
			print(traceback.format_exc())
			print('Disconnected, waiting...')
			self.start_connecting()
	
	def send_message(self, message):
		try:
			self.client_soc.send((message + "\n").encode('utf-8'))
		except Exception as e:
			print(e)
			print("Client disconnected!")