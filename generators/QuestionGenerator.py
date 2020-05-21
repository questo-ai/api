# -*- coding: utf-8 -*-
import abc, six
import zmq
import random

# @six.add_metaclass(abc.ABCMeta)

class QuestionGenerator(object, metaclass=abc.ABCMeta):
	def __init__(self):
		self.quiz = []
		port = "5556"
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REP)
		self.socket.bind("tcp://*:%s" % port)
		print("Serving {} on tcp://*:{}".format(type(self).__name__, port))
		self.serve()

	@abc.abstractmethod
	def generate_questions(self, data):
		'''
		Method called by the generator class
		to generate questions given text
		'''
		raise NotImplementedError("need to create child class and define :generateQuiz()")

	@abc.abstractmethod
	def validate(self, qas):
		'''
		Method called internally to check
		the validity of generated content
		'''
		raise NotImplementedError("need to define :nonsensicality() for internal use")

	def shuffle(self, naive_qas):
		'''
		Shuffles input qas smartly to prevent
		the following from appearing together:
		1. Questions with similar answers
		2. Questions based on the same sentence
		

		# before
		e.g qas = [{'question': 'Who killed himself in 1945?',
				'answer': 'Hitler',
				'sentence': 'Hitler killed himself in 1945.'}...]

		# after
		e.g qas = [{'sentence': 'Hitler killed himself in 1945.',
				'questions': [{'question': 'Who killed himself in 1945?'
							'answer': 'Hitler'}...]}...]

		note: if iOS wants sents, remove popping
		of sentence from qa.
		'''
		qas = []
		unique_sents = []
		for qa in naive_qas:
			try:
				i = unique_sents.index(qa['sentence'])
				qas[i]['questions'].append({k:v for k,v in qa.items() if k != 'sentence'})
			except ValueError:
				unique_sents.append(qa['sentence'])
				qas.insert(
					len(qas),
					{'sentence': qa['sentence'],
					'questions': [{k:v for k,v in qa.items() if k != 'sentence'}]}
				)
		unsimilars = [s['questions'][0] for s in qas if len(s['questions']) == 1]	
		similars = [s for s in qas if len(s['questions']) > 1]
		if unsimilars == None:
			unsimilars = []
		elif similars == None:
			similars = []
			unsimilars = random.shuffle(unsimilars)
		for sim in similars:
			questions = sim['questions']
			unsimilars.insert(0, questions[0])
			unsimilars.append(questions[1])
		return unsimilars

	def serve(self):
		while True:
			arguments = self.socket.recv_json()
			questions = self.shuffle(self.generate_questions(arguments))
			self.socket.send_json(questions)
