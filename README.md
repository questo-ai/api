# questo
<img src="./Questo Network Diagram.svg">

## design principles
- Only tabs.
- Readable and self-documented code is a must. Use comments
- Khush, if you use one-character variables... 🔪
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
