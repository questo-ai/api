# -*- coding: utf-8 -*-
import abc, six
import zmq
import random
import spacy


class SpacyServer(object):
	def __init__(self):
		port = "5556"
		print("Loading large model...")
		self.nlp = spacy.load("en_core_web_lg")
		print("Large model loaded.")
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REP)
		self.socket.bind("tcp://*:%s" % port)
		print("Serving {} on tcp://*:{}".format(type(self).__name__, port))
		self.serve()

	def tokenize(self, data):
		'''
		Method to tokenize text and dump its bytes
		representation into hexadecimal for
		transmission.
		'''
		text = data['text']
		doc = self.nlp(text)
		doc_bytes_as_hex = doc.to_bytes().hex()
		data = {"hex_encoded_doc": doc_bytes_as_hex}

		'''
		reconstructed by:
		retrieved = data['hex_encoded_doc']
		nlp('').from_bytes(bytes.fromhex(retrieved))
		'''

		return data

	def serve(self):
		while True:
			arguments = self.socket.recv_json()
			tokenized_text = self.tokenize(arguments)
			self.socket.send_json(tokenized_text)
SpacyServer()