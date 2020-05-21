import time
import zmq
import socket

REQUEST_TIMEOUT = 10000
REQUEST_RETRIES = 3

from spacy.lang.en import English
nlp = English()

def run(data, slave):
	'''
	Standard function for interfacing with a ZMQ server.
	data: (dict)
	slave: (string) IP of server
	'''
	print("running")
	context = zmq.Context()
	client = context.socket(zmq.REQ)

	poll = zmq.Poller()
	poll.register(client, zmq.POLLIN)

	ip = socket.gethostbyname(slave) # left here to convert localhost to 127.0.0.1 just in case
	SERVER_ENDPOINT = "tcp://{}:5556".format(ip)
	client.connect(SERVER_ENDPOINT)

	sequence = 0
	retries_left = REQUEST_RETRIES
	
	while retries_left:
		sequence += 1
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
				retries_left -= 1
				if retries_left == 0:
					print("E: {} seems to be offline, abandoning".format(name))
					break
				print("I: Reconnecting and resending ({})".format(sequence))
				# Create new connection
				client = context.socket(zmq.REQ)
				client.connect(SERVER_ENDPOINT)
				poll.register(client, zmq.POLLIN)
				client.send_json(data)

def example():
	text = """I like apples."""
	data = {'text': text}
	print('yeet')

	results = run(data, '34.87.95.66')
	print('yeet')
	retrieved = results['hex_encoded_doc']

	reconstructed_doc = nlp('').from_bytes(bytes.fromhex(retrieved))

	return reconstructed_doc

print(example())
