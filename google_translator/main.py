import datetime
import os, sys
import socket
import argparse
import threading
import configparser
import time

from googletrans import Translator
from script_server import Server

class GoogleTranslate:

	def __init__(self, filename_config) -> None:
		
		config = configparser.ConfigParser()
		config.read(filename_config, encoding='utf-8')

		# Load configuration file
		self.connection_port = config['CONNECTION'].getint('port')
		self.translate_service_url = config['TRANSLATE'].get('service_url')
		self.translate_lang_source = config['TRANSLATE'].get('source')
		self.translate_lang_dest = config['TRANSLATE'].get('dest')

		print(self.connection_port)
		print(self.translate_lang_source)
		print(self.translate_lang_dest)
		#self.translate_lang_to = config['TRANSLATE'].get('to')

		self.server = Server("localhost", self.connection_port, self)
		# self.translator = Translator(service_urls=[self.translate_service_url])
		self.translator = Translator()
		self.translator.raise_Exception = True

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

	gt = GoogleTranslate(args.config)
