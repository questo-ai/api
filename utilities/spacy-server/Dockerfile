FROM ubuntu:latest
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3.6 python3-pip

COPY ./requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
RUN python3 -m spacy download en_core_web_lg
ENV SPACY_WARNING_IGNORE=W008 

COPY ./ /app
WORKDIR /app
ENTRYPOINT ["python3", "spacy_server.py"]