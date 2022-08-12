import datetime
import os, sys
import socket
import argparse
import threading
import configparser
import time

from script_server import Server
from bot import Bot

class ChattingYamamoto:

	def __init__(self, filename_config) -> None:
		
		config = configparser.ConfigParser()
		config.read(filename_config, encoding='utf-8')

		# Load configuration file
		self.connection_port = config['CONNECTION'].getint('port')
		print(self.connection_port)

		self.bot = Bot()
		self.server = Server("localhost", self.connection_port, self)

		t1 = threading.Thread(target=self.server.start_connecting, args=())
		t1.setDaemon(True)
		t1.start()
		self.loop()
	
	def translate(self, sentence):
		translated = self.translator.translate(sentence, src=self.translate_lang_source, dest=self.translate_lang_dest)

		print("FROM: " + sentence)
		print("DEST: " + translated.text)
		print()

		return translated.text

	def loop(self):
		while True:
			time.sleep(0.5)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Google translate client module')
	parser.add_argument('-c', '--config', help='config filename', default='config-sample.txt')

	args = parser.parse_args()

	cy = ChattingYamamoto(args.config)
