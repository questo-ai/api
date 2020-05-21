from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification
from pathlib import Path
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
import pandas as pd
import os
import sys
import numpy as np
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.optimization import BertAdam

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
	"""BERT model for classification.
	This module is composed of the BERT model with a linear layer on top of
	the pooled output.
	Params:
		`config`: a BertConfig class instance with the configuration to build a new model.
		`num_labels`: the number of classes for the classifier. Default = 2.
	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
			types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
			a `sentence B` token (see BERT paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
			with indices selected in [0, ..., num_labels].
	Outputs:
		if `labels` is not `None`:
			Outputs the CrossEntropy classification loss of the output with the labels.
		if `labels` is `None`:
			Outputs the classification logits of shape [batch_size, num_labels].
	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
	num_labels = 2
	model = BertForSequenceClassification(config, num_labels)
	logits = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config, num_labels=2):
		super(BertForMultiLabelSequenceClassification, self).__init__(config)
		self.num_labels = num_labels
		self.bert = BertModel(config)
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
		_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		if labels is not None:
			loss_fct = BCEWithLogitsLoss()
			loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
			return loss
		else:
			return logits
		
	def freeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = False
	
	def unfreeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = True

class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()
	
	def get_test_examples(self, data_dir, data_file_name, size=-1):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError() 

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, labels=None):
		"""Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			labels: (Optional) [string]. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.labels = labels


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_ids):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_ids = label_ids

class MultiLabelTextProcessor(object):
	
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.labels = None

	def _create_examples(self, df, set_type, labels_available=True):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, row) in enumerate(df.values):
			guid = i
			text_a = row[0]
			if labels_available:
				label = [0,0]
				r = row[1]
				label[r] = 1
			else:
				label = []
			examples.append(
				InputExample(guid=guid, text_a=text_a, labels=label))
		return examples

class BERTRatingModel(object):
	def __init__(self):
		print("Loading BERT Rating model...")
		BERT_PRETRAINED_PATH = Path('uncased_L-12_H-768_A-12/')
		DATA_PATH=Path('data/')
		DATA_PATH.mkdir(exist_ok=True)
		PYTORCH_PRETRAINED_BERT_CACHE = BERT_PRETRAINED_PATH/'cache/'
		PYTORCH_PRETRAINED_BERT_CACHE.mkdir(exist_ok=True)
		
		PATH=Path('data/tmp')
		PATH.mkdir(exist_ok=True)

		CLAS_DATA_PATH=PATH/'class'
		CLAS_DATA_PATH.mkdir(exist_ok=True)

		args = {
			"train_size": -1,
			"val_size": -1,
			"full_data_dir": DATA_PATH,
			"data_dir": PATH,
			"task_name": "question-rating",
			"bert_model": BERT_PRETRAINED_PATH,
			"output_dir": CLAS_DATA_PATH/'output',
			"max_seq_length": 100,
			"do_train": True,
			"do_eval": True,
			"do_lower_case": True,
			"train_batch_size": 32,
			"eval_batch_size": 32,
			"learning_rate": 3e-5,
			"num_train_epochs": 3.0,
			"warmup_proportion": 0.1,
			"no_cuda": True,
			"local_rank": -1,
			"seed": 42,
			"gradient_accumulation_steps": 1,
			"optimize_on_cpu": False,
			"fp16": False,
			"loss_scale": 128
		}

		label_list = ['bad', 'good']
		self.label_list = label_list
		num_labels = len(label_list)

		# Setup GPU parameters
		if args["local_rank"] == -1 or args["no_cuda"]:
			device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
			n_gpu = torch.cuda.device_count()
		#     n_gpu = 1
		else:
			torch.cuda.set_device(self.args['local_rank'])
			device = torch.device("cuda", args['local_rank'])
			n_gpu = 1
			# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
			torch.distributed.init_process_group(backend='nccl')

		output_model_file = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "finetuned_pytorch_model.bin")
		if args["local_rank"] == -1 or args["no_cuda"]:
			self.model_state_dict = torch.load(output_model_file, map_location='cpu')
		else:
			self.model_state_dict = torch.load(output_model_file)
		self.model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels=num_labels, state_dict=self.model_state_dict);
		self.model.to(device);
		self.args = args
		self.device = device
		self.tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])
		print("Model loaded.")

	def predict(self, sents_and_questions):
		questions = [InputExample(str(i), q, None, []) for i, q in enumerate(sents_and_questions)]
		input_data = [{'id': q.guid, 'question': q.text_a} for q in questions]

		features = self.convert_examples_to_features(questions, self.label_list, self.args['max_seq_length'], self.tokenizer)
		
		all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

		test_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
		
		# Run prediction for full data
		test_sampler = SequentialSampler(test_dataset)
		test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=self.args['eval_batch_size'])
		
		all_logits = None
		
		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0
		for step, batch in enumerate(test_dataloader):
			input_ids, input_mask, segment_ids = batch
			input_ids = input_ids.to(self.device)
			input_mask = input_mask.to(self.device)
			segment_ids = segment_ids.to(self.device)

			with torch.no_grad():
				logits = self.model(input_ids, segment_ids, input_mask)
				logits = logits.sigmoid()

			if all_logits is None:
				all_logits = logits.detach().cpu().numpy()
			else:
				all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
				
			nb_eval_examples += input_ids.size(0)
			nb_eval_steps += 1
		
		return [{"question": input_data[i]['question'], "rating": self.label_list[np.argmax(ex)]} for i, ex in enumerate(all_logits)]

	def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
		"""Loads a data file into a list of `InputBatch`s."""

		label_map = {label : i for i, label in enumerate(label_list)}

		features = []
		for (ex_index, example) in enumerate(examples):
			tokens_a = tokenizer.tokenize(example.text_a)

			tokens_b = None
			if example.text_b:
				tokens_b = tokenizer.tokenize(example.text_b)
				# Modifies `tokens_a` and `tokens_b` in place so that the total
				# length is less than the specified length.
				# Account for [CLS], [SEP], [SEP] with "- 3"
				_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
			else:
				# Account for [CLS] and [SEP] with "- 2"
				if len(tokens_a) > max_seq_length - 2:
					tokens_a = tokens_a[:(max_seq_length - 2)]

			# The convention in BERT is:
			# (a) For sequence pairs:
			#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
			#  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
			# (b) For single sequences:
			#  tokens:   [CLS] the dog is hairy . [SEP]
			#  type_ids: 0   0   0   0  0     0 0
			#
			# Where "type_ids" are used to indicate whether this is the first
			# sequence or the second sequence. The embedding vectors for `type=0` and
			# `type=1` were learned during pre-training and are added to the wordpiece
			# embedding vector (and position vector). This is not *strictly* necessary
			# since the [SEP] token unambigiously separates the sequences, but it makes
			# it easier for the model to learn the concept of sequences.
			#
			# For classification tasks, the first vector (corresponding to [CLS]) is
			# used as as the "sentence vector". Note that this only makes sense because
			# the entire model is fine-tuned.
			tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
			segment_ids = [0] * len(tokens)

			if tokens_b:
				tokens += tokens_b + ["[SEP]"]
				segment_ids += [1] * (len(tokens_b) + 1)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)

			# The mask has 1 for real tokens and 0 for padding tokens. Only real
			# tokens are attended to.
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			padding = [0] * (max_seq_length - len(input_ids))
			input_ids += padding
			input_mask += padding
			segment_ids += padding

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length
			
			labels_ids = []
			for label in example.labels:
				labels_ids.append(float(label))

			features.append(
					InputFeatures(input_ids=input_ids,
								  input_mask=input_mask,
								  segment_ids=segment_ids,
								  label_ids=labels_ids))
		return features

	def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
		"""Truncates a sequence pair in place to the maximum length."""

		# This is a simple heuristic which will always truncate the longer sequence
		# one token at a time. This makes more sense than truncating an equal percent
		# of tokens from each, since if one sequence is very short then each token
		# that's truncated likely contains more information than a longer sequence.
		while True:
			total_length = len(tokens_a) + len(tokens_b)
			if total_length <= max_length:
				break
			if len(tokens_a) > len(tokens_b):
				tokens_a.pop()
			else:
				tokens_b.pop()
