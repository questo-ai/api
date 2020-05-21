import pytest

import time
import zmq
import socket

from .GFGenerator import GFQuestions

d = {"text": "The Suez Crisis, or the Second Arab–Israeli War, also named the Tripartite Aggression in the Arab world and Operation Kadesh or Sinai War in Israel, was an invasion of Egypt in late 1956 by Israel, followed by the United Kingdom and France. The aims were to regain Western control of the Suez Canal and to remove Egyptian President Gamal Abdel Nasser, who had just nationalized the canal. After the fighting had started, political pressure from the United States, the Soviet Union and the United Nations led to a withdrawal by the three invaders. The episode humiliated the United Kingdom and France and strengthened Nasser."}

@pytest.fixture
def instance():
	g = GFQuestions(testing=True, model="en_core_web_sm")
	yield g

def test_questions(instance):
	response = instance.generate_questions(d)
	print(response)
	assert response != None
