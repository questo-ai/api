# -*- coding: utf-8 -*- 
try:
	from .QuestionGenerator import QuestionGenerator
	from .QuestionRater import BERTRatingModel
except (ModuleNotFoundError, ImportError):
	from QuestionGenerator import QuestionGenerator
	from QuestionRater import BERTRatingModel
import random
import spacy

class TemplatedQuestions(QuestionGenerator):

	def __init__(self, model="en_core_web_lg", testing=False, local=False):
		if not local:
			print("Loading and building BERT model...")
			self.rating_model = BERTRatingModel()
			print("BERT loaded. Loading {}...".format(model))
		self.local = local
		self.nlp = spacy.load(model)
		# threshold for how similar title can be to 
		# a potential answer-phrase
		self.title_similarity = 0.9
		print("{} loaded.".format(model))
		if not testing:
			super().__init__()


	def merge_ents(self, doc):
		spans = list(doc.ents) + list(doc.noun_chunks)
		spans = self.filter_spans(spans)
		with doc.retokenize() as retokenizer:
			for span in spans:
				retokenizer.merge(span)
		return doc

	def chunk(self, doc):
		chunks = []
		used_sents = []
		used_ents = []
		for ent in doc.ents:
		# for ent in filter(lambda e: e.label_ == 'DATE', doc.ents):
			if ent.sent not in used_sents:
				# primary chunking logic
				array = []
				for l in ent.sent.root.lefts: array.extend(l.subtree)
				array.append(ent.sent.root)
				right = ent.sent.root.rights
				right_subtree = next(right).subtree
				array.extend(right_subtree)
				while ent[0].text not in ''.join([t.text_with_ws for t in array]):
					try:
						array.extend(next(right).subtree)
					except StopIteration:
						break

				# make sure all punctuation is added
				# e.g. "In 1775, he was appointed Grand Master of the Landesloge of Germany (Zinnendorf system"
				# --> "In 1775, he was appointed Grand Master of the Landesloge of Germany (Zinnendorf system)."
				while array[-1].i+1 < len(doc) and doc[array[-1].i+1].is_punct:
					array.append(doc[array[-1].i+1])
				used_sents.append(ent.sent)
				used_ents.append(ent)
				text_array = [t.text_with_ws for t in array]

				# handles cases where chunking mid-sentence
				if not array[-1].is_punct:
					text_array[-1] = array[-1].text
					text_array.append(". ")
				elif array[-1].text in [',', ';', ':']:
					text_array[-1] = '. '
				chunks.append("".join(text_array))
		return chunks

	def generate_questions(self, data):
		text = data['text']
		title = self.nlp(data['title'])
		subject = self.nlp(data['subject'])
		doc = self.nlp(text)
		doc = self.merge_ents(self.nlp("".join(self.chunk(doc))))
		questions = []
		questions.extend(self.extract_date_relations(doc, title, subject))
		questions.extend(self.extract_person_relations(doc, title, subject))
		questions.extend(self.extract_location_relations(doc, title, subject))

		if self.local:
			return questions
		else:
			return self.validate(questions) 

	def validate(self, qas):
		if len(qas) == 0:
			return []
		else:
			rated = self.rating_model.predict([qa["question"] for qa in qas])
			return [qas[i] for i, r in enumerate(rated) if r['rating'] == 'good']

	def extract_date_relations(self, doc, title, subject):
		# Merge entities and noun chunks into one token
		used_dates = []
		relations = []
		for date in filter(lambda w: w.ent_type_ == "DATE" and w.text not in used_dates, doc):
			if self.nlp(date.text).similarity(title) > self.title_similarity:
				continue
			if date.dep_ == "pobj" and date.head.dep_ == "prep":
				used_dates.append(date.text)
				verb = date.head.head.head
				counter = 0
				while verb.head != verb and counter < 10:
					verb = verb.head
					counter += 1
				
				auxpass = [w for w in verb.children if w.dep_ is 'auxpass']
				xcomp = [w for w in verb.children if w.dep_ is 'xcomp']
				nsubjpass = [w for w in verb.children if w.dep_ is 'nsubjpass']
				children = [(w,w.dep_) for w in verb.children]
				nsubj = [w for w in verb.children if w.dep_ is 'nsubj']
				dobj = [w for w in verb.children if w.dep_ is 'dobj']
				pobj = [w for w in verb.children if w.dep_ is 'pobj']
				prep = [w for w in verb.children if w.dep_ is 'prep']
				agent = [w for w in verb.children if w.dep_ is 'agent']
				attr = [w for w in verb.children if w.dep_ is 'attr']
				
				if date.head.lower_ in ['from', 'until', 'after', 'following']:
					# "The Suez Canal was closed from 1945 to 2004."
					if len(nsubj) > 0 and len(dobj) > 0:
						array = [date.head.text.capitalize(), 'when did', nsubj[0].text, verb.lemma_]
						dobj_subtree = [child for child in dobj[0].subtree]
						array.extend([w.text for w in dobj_subtree])
					elif len(nsubjpass) > 0:
						array = [date.head.text.capitalize(), 'when', auxpass[0].text, nsubjpass[0].text, verb.text]
						if len(agent) > 0:
							agent_subtree = [child for child in agent[0].subtree]
							array.extend([w.text for w in agent_subtree])
					else: 
						continue
					relations.append({'sentence': date.sent.text, 'question': ' '.join(array)+"?", 'answer': date.text})
				elif len(nsubj) > 0 and len(prep) > 0:
					if nsubj[0].pos_ == 'ADJ' or prep[0].lower_ != "in":
						continue
					# "In 2012, alcohol laws in Alabama changed."
					array = ["When did"]
					nsubj_subtree = [child for child in nsubj[0].subtree]
					array.extend([w.text for w in nsubj_subtree])
					if len(dobj) > 0:
						array.append(verb.lemma_)
						dobj_subtree = [child for child in dobj[0].subtree]
						array.extend([w.text for w in dobj_subtree])
					else:
						array.append(verb.lemma_)
					relations.append({'sentence': date.sent.text, 'question': ' '.join(array)+"?", 'answer': date.text})
				elif len(auxpass) > 0 and len(nsubjpass) > 0: 
					## "The Suez Canal was closed from 1920."
					nsubjpass_subtree = [child for child in nsubjpass[0].subtree]
					if date.head.lower_ in ['in', 'around', 'after', 'following']:
						array = ['When', auxpass[0].text]
						array.extend([w.text for w in nsubjpass_subtree])
						array.append(verb.text)
						relations.append({'sentence': date.sent.text, 'question': ' '.join(array)+"?", 'answer': date.text})
				elif date.head.text in ['of']:
					continue
				elif verb:
					left_roots = [w for w in verb.lefts if w.dep_ == "nsubj"]
					left_phrase = [" ".join([t.text for t in root.subtree]) for root in left_roots]
					right_roots = [w for w in verb.rights if((w.pos_ is not "PUNCT") and not "DATE" in [w.ent_type_ for w in w.children])]
					right_phrase = [" ".join([t.text for t in root.subtree]) for root in right_roots]
					if len(right_phrase) > 0 and len(left_phrase) > 0:
						left = ' '.join(left_phrase)
						right = ' '.join(right_phrase)
						question = ' '.join(["When did", str(left), str(verb.lemma_), str(right) + "?"])
						relations.append({'sentence': date.sent.text, 'question': question.replace('  ',' '), 'answer': date.text})
		return relations

	def extract_person_relations(self, doc, title, subject):
		# Merge entities and noun chunks into one token
		used_people = []
		relations = []
		for person in filter(lambda w: w.ent_type_ == "PERSON" and w.text not in used_people, doc):
			if self.nlp(person.text).similarity(title) > self.title_similarity:
				continue
			if person.dep_ == "nsubj" or person.dep_ == "nsubjpass":
				used_people.append(person.text)
				verb = person.head.head
				has_conj = True
				auxpass = [w for w in verb.children if w.dep_ is 'auxpass']
				xcomp = [w for w in verb.children if w.dep_ is 'xcomp']
				nsubj = [w for w in verb.children if w.dep_ in ['nsubj', 'nsubjpass']]
				if len(auxpass) > 0 and len(nsubj) > 0:
					# SENT: "Goh Chok Tong was passed the reins of leadership by Lee Kuan Yew in 1990."
					# --> "Who was passed the reins of leadership by Lee Kuan Yew in 1990?", "Goh Chok Tong"
					right_roots = [w for w in verb.rights if((w.pos_ is not "PUNCT") and not "person" in [w.ent_type_ for w in w.children])]
					right_phrase = [" ".join([t.text for t in root.subtree]) for root in right_roots]
					array = ['Who', auxpass[0].text]
					array.append(verb.text)
					array.extend(right_phrase)
					relations.append({'sentence': person.sent.text, 'question': ' '.join(array)+"?", 'answer': person.text})
				elif verb:
					if verb.dep_ == 'ROOT':
						# SENT: "In 1990, Lee Kuan Yew passed the reins of leadership to Goh Chok Tong."
						# --> "Who passed the reins of leadership to Goh Chok Tong?", "Lee Kuan Yew"
						right_roots = [w for w in verb.rights if((w.pos_ is not "PUNCT") and not "person" in [w.ent_type_ for w in w.children] and w.dep_ not in ["cc", "conj"])]
						right_phrase = [" ".join([t.text for t in root.subtree]) for root in right_roots]
						if len(right_phrase) > 0:
							right = ' '.join(right_phrase)
							question = ' '.join(["Who", str(verb), str(right) + "?"])
							relations.append({'sentence': person.sent.text, 'question': question.replace('  ',' '), 'answer': person.text})
		return relations

	def extract_location_relations(self, doc, title, subject):
		# Merge entities and noun chunks into one token
		used_locations = []
		relations = []
		for location in filter(lambda w: w.ent_type_ == "GPE" and w.text not in used_locations, doc):
			if self.nlp(location.text).similarity(title) > self.title_similarity:
				continue
			if location.dep_ == "pobj" and location.head.dep_ == "prep":
				used_locations.append(location.text)
				verb = location.head.head
				has_conj = True
				aux = [w for w in verb.children if w.dep_ is 'aux']
				xcomp = [w for w in verb.children if w.dep_ is 'xcomp']
				nsubj = [w for w in verb.children if w.dep_ is 'nsubjpass']
				if verb.text in ['is', 'was', 'were']: # state of being adjective
					left_roots = [w for w in verb.lefts if w.dep_ == "nsubj"]
					left_phrase = ' '.join([" ".join([self.lowerize(t) for t in root.subtree]) for root in left_roots])
					right_roots = [w for w in verb.rights if w.dep_ in ['acomp', 'amod']]
					right_phrase = ' '.join([" ".join([self.lowerize(t) for t in root.subtree]) for root in right_roots])
					question = ' '.join(["Where", str(verb), left_phrase, right_phrase+"?"])
					relations.append({'sentence': location.sent.text, 'question': question.replace('  ',' '), 'answer': location.text})
				else:
					left_roots = [w for w in verb.lefts if w.dep_ == "nsubj"]
					left_phrase = [" ".join([self.lowerize(t) for t in root.subtree]) for root in left_roots]
					if len(left_phrase) > 0:
						left = ' '.join(left_phrase)
						question = ' '.join(["Where did", str(left), str(verb.lemma_)+"?"])
						relations.append({'sentence': location.sent.text, 'question': question.replace('  ',' '), 'answer': location.text})
		return relations

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

	def lowerize(self, tok):
		if tok.is_sent_start: 
			return tok.lower_
		else:
			return tok.text

if __name__ == "__main__":
	TemplatedQuestions(model="en_core_web_lg")
