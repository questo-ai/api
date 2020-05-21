# -*- coding: utf-8 -*-
import abc, six
import zmq
import launch

class KeywordExtraction(object, metaclass=abc.ABCMeta):
	def __init__(self):
		self.quiz = []
		port = "5556"
		self.beta = 0.7
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REP)
		print("loading embedding_distributor")
		self.embedding_distributor = launch.load_local_embedding_distributor()
		print("loading pos_tagger")
		self.pos_tagger = launch.load_local_corenlp_pos_tagger()
		self.socket.bind("tcp://*:%s" % port)
		print("Serving {} on tcp://*:{}".format(type(self).__name__, port))
		self.serve()
    
	def generate_keywords(self, text):
		print("generating")
		return launch.extract_keyphrases(
			self.embedding_distributor,
			self.pos_tagger,
			text,
			10,
			'en',
			self.beta
		)

	def serve(self):
		while True:
			data = self.socket.recv_json()
			print(data)
			text = data['text']
			questions = self.generate_keywords(text)
			self.socket.send_json(questions)
KeywordExtraction()
