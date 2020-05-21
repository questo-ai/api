# -*- coding: utf-8 -*- 
import json, sys, os
try:
	from .QuestionGenerator import QuestionGenerator
except (ModuleNotFoundError, ImportError):
	from QuestionGenerator import QuestionGenerator
from rake_nltk import Metric, Rake 
import zmq
import socket
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from copy import deepcopy

SWISSCOM_KEYWORD_URL = ''
SWISSCOM_KEYWORD_PORT = '5556'
SWISSCOM_KEYWORD_IP = socket.gethostbyname(SWISSCOM_KEYWORD_URL)

class GFQuestions(QuestionGenerator):
	def __init__(self, model="en_core_web_lg", testing=False):
		print("Loading {}...".format(model))
		self.nlp = spacy.load(model)
		self.lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
		self.title_similarity = 0.90
		print("{} loaded.".format(model))
		if not testing:
			super().__init__()


	def generate_questions(self, data):
		text = data['text']
		title = self.nlp(data['title'])
		subject = self.nlp(data['subject'])
		doc = self.nlp(text)
		generated = []
		ranking_metrics = [Metric.WORD_DEGREE]
		sentences = [s.text for s in doc.sents]
		phrases_swisscom = self.generate_keywords(text)
		phrases = []
		# EG: [['egyptian president gamal abdel nasser', 'suez canal', 'israeli war', 'arab world', 'egypt', 'suez crisis', 'soviet union', 'nasser', 'tripartite aggression', 'israel'], [1.0, 0.8614248633384705, 0.8030354976654053, 0.7896698713302612, 0.811191737651825, 0.8514521718025208, 0.6438262462615967, 0.8813737034797668, 0.5584405660629272, 0.7795075178146362], [['nasser'], ['canal'], [], [], [], [], [], ['egyptian president gamal abdel nasser'], [], []]]

		if phrases_swisscom == None:	
			# fallback option just in case swisscom isn't working
			for metric in ranking_metrics:
				r = Rake(ranking_metric=metric, min_length=1, max_length=5)
				
				# Extraction given the sentences as a list of strings.
				r.extract_keywords_from_sentences(sentences)
				
				# To get keyword phrases ranked highest to lowest and strip out the last half. 
				keywords = r.get_ranked_phrases()
				keywords = keywords[0: round(len(keywords)*0.5)]
				phrases.extend(keywords)
		else:
			print(phrases_swisscom)
			phrases = sorted([(p, phrases_swisscom[1][i]) for i, p in enumerate(phrases_swisscom[0])
				if title.similarity(self.nlp(p)) < self.title_similarity and subject.similarity(self.nlp(p)) < self.title_similarity],
				key=lambda x: x[1],
				reverse=True)
			phrases = [p for p, s in phrases if s > 0.5]

		generated = []
		sentences_used = {s:0 for s in doc.sents}
		phrases_used = []

		for tok in doc:
			for phrase in phrases:
				tok_sent_i = tok.i - tok.sent.start
				tok_sent_end = tok_sent_i+len(phrase.split())
				same_len = tok.sent[tok_sent_i:tok_sent_end]
				if [t.lower_ for t in same_len] == phrase.lower().split():
					similarity = [self.nlp(phrase).similarity(p) for p in phrases_used] # figure out empty vector
					if sentences_used[tok.sent] < 3 and max(similarity, default=0.0) < self.title_similarity:
						toks_with_ws = [token.text_with_ws for token in tok.sent]
						long_gap_toks = deepcopy(toks_with_ws)
						for i in range(tok_sent_i, tok_sent_end):
							long_gap_toks[i] = '_'*len(tok.sent[i])
							if i == tok_sent_i:
								toks_with_ws[i] = '_____'
							else:
								toks_with_ws[i] = ''
						long_gap_toks[tok_sent_end-1] = long_gap_toks[tok_sent_end-1] + tok.sent[tok_sent_end-1].whitespace_
						toks_with_ws[tok_sent_end-1] = toks_with_ws[tok_sent_end-1] + tok.sent[tok_sent_end-1].whitespace_
						pair = {
							"question": "".join(long_gap_toks),
							"answer": "".join([t.text_with_ws for t in same_len]),
							"sentence": tok.sent.text,
							"short_gap": "".join(toks_with_ws)
						}
						sentences_used[tok.sent] += 1
						phrases_used.append(self.nlp(phrase))
						generated.append(pair)

		return generated

	@staticmethod
	def generate_keywords(text):
		REQUEST_TIMEOUT = 3000
		REQUEST_RETRIES = 3
		context = zmq.Context()
		client = context.socket(zmq.REQ)

		poll = zmq.Poller()
		poll.register(client, zmq.POLLIN)

		ip = SWISSCOM_KEYWORD_IP
		SERVER_ENDPOINT = "tcp://{}:{}".format(ip, SWISSCOM_KEYWORD_PORT)
		# print(SERVER_ENDPOINT)
		client.connect(SERVER_ENDPOINT)

		data = {'text': text}

		sequence = 0
		retries_left = REQUEST_RETRIES
		while retries_left:
			sequence = sequence + 1
			client.send_json(data)
			expect_reply = True
			
			while expect_reply:
				socks = dict(poll.poll(REQUEST_TIMEOUT))
				if socks.get(client) == zmq.POLLIN:
					try:
						reply = client.recv_json()
						retries_left = REQUEST_RETRIES
						expect_reply = False
						return reply
					except TypeError:
						print("E: Malformed reply from server: ")
						print(reply)
				else:
					print("W: No response from server, retrying...")
					client.setsockopt(zmq.LINGER, 0)
					client.close()
					poll.unregister(client)
					retries_left = retries_left - 1
					if retries_left == 0:
						print("E: Server seems to be offline, abandoning")
						expect_reply = False
						return None
						break
					print("I: Reconnecting and resending ({})".format(sequence))
					# Create new connection
					client = context.socket(zmq.REQ)
					client.connect(SERVER_ENDPOINT)
					poll.register(client, zmq.POLLIN)
					client.send_json(data)

	def filter_spans(self, spans):
		# Filter a sequence of spans so they don't contain overlaps
		get_sort_key = lambda span: (span.end - span.start, span.start)
		sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
		result = []
		seen_tokens = set()
		for span in sorted_spans:
			if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
				result.append(span)
				seen_tokens.update(range(span.start, span.end))
		return result

	@classmethod
	def validate(self, question, answer):
		pass

if __name__ == "__main__":
	GFQuestions(model="en_core_web_lg")
