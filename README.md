# questo
<img src="./Questo Network Diagram.svg">

## A Note

After more than three years of working on Questo, the three of us have decided it's time to close the chapter on this part of our lives. Questo has been an amazing experience — we're grown together in ways we never would've thought possible, and become better developers for it. We'll be releasing a writeup shortly to accompany this, but for the time being here is all our source code for the backend. 

The `generators` directory contains all the actual question generation technology, including our novel algorithm for generating "template" style (e.g Who, What, Where) questions. 

The `research` folder contains a utility for remotely serving `spaCy`. This was very useful during development, because otherwise we'd have to keep reloading the model, wasting lots of time in the process. 

The `scripts` folder contains the shell-script used to push our Docker images to the Kubernetes clusters hosted on Google Cloud.

The `utilities` folder contains the files needed to build the `spaCy` server's Docker image, as well as the Docker image for a keyword extractor. 

Please feel free to reach out if you'd like to discuss the project further!

## design principles
- Only tabs.
- Readable and self-documented code is a must. Use comments
- ZMQ microservices are used for internal messaging, speeeeed baby
- meaningful commit messages when possible 🤘🏼
- no unilateral changes to API without conferring with iOS

## how we split stuff up
[Network Diagram](https://www.draw.io/?lightbox=1&highlight=0000ff&edit=_blank&layers=1&nav=1#G1ONXd7Tv7sdLaXIYwSRYsI3VG4YXfgXIA)

## deployment
1. travis is triggered if `TEMPLATE_ONLY` or `GAPFILL_ONLY` are in commit heads, else, if commits are made to `stable`.
2. travis internally connects to gcloud, from which the latest versions of a generator are pulled, rebuilt & tested.
3. pytest is used to test all generators and the API layer.
4. travis will optionally deploy _iff_ in branch `stable`.

## how to develop
1. clone this, use conda to create an environment for generators.
2. run `python -m spacy download en_core_web_sm` for easy local development.
3. if doing rigorous spaCy NLP, use the remote spaCy ZMQ service. refer to `utilities/spacy-server`.
4. refer to design principles.
