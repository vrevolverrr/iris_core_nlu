# NLU Core For Project Iris

This is a component of Project Iris, a Windows based personal assistant written in Python, Dart and NodeJS.
The intent classification model is based on Tensorflow embeddings trained from intent classification data. Entity extraction uses the Spacy NLP pipeline with sklearn CRF Enitity Extraction pipeline.

### Usage

Project Iris uses Python Rasa NLU with Tensorflow backend to train models for intent classification and entitity extraction.
Intent Classification models are seperate from Entitiy Extraction models.

This module is desinged to work as a subprocess where I/O is done through stdin and stdout.

To start the process :
```sh
$ python ./Model.py start
```

To train all models :
```sh
$ python ./Model.py train-models
```

To only train intent classification model :
```sh
$ python ./Model.py train-intents
```

To only train entity extraction models :
```sh
$ python ./Model.py train-intents
```

#### Training Data
Intent classification training data is stored under ./data/intents.json
Entity extraction training data is stored under ./data/intent_entities/{intent_name}.json

Intent classification training data must not contain any data on entities. Entity extraction information must only be declared in a seperate training data JSON file.

All training data must follow the Rasa NLU JSON training data format. 
https://legacy-docs.rasa.com/docs/nlu/0.15.1/dataformat/
