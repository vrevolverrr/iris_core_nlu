import os
import sys
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from rasa_nlu import config
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.model import Trainer, Interpreter
from rasa_nlu.training_data import load_data
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
if type(tf.contrib) != type(tf): tf.contrib._warning = None
from src.Intent import Intent
from src.Interprocess import InterprocessHandler

class NLUModel(InterprocessHandler):
    """ Instantiates the model """
    def __init__(self, config_path, models_path, data_path):
        config_file = config.load(config_path)
        self.builder = ComponentBuilder(use_cache=True)
        self.trainer = Trainer(config_file, self.builder)
        self.intent_model = None
        self.entity_models = dict()
        self.models_path = models_path
        self.data_path = data_path

    """ Trains intent classification model """
    def train_intents_model(self):
        intent_training_data = load_data(os.path.join(self.data_path, 'intents.json'))
        self.trainer.train(intent_training_data)
        self.trainer.persist(self.models_path, fixed_model_name='intents')

    """ Trains entity extraction models """
    def train_entitiy_models(self, fixed_intent_name = None):
        entities_data_path = os.path.join(self.data_path, "intent_entities")
        
        if fixed_intent_name is None:
            for entities_data in os.listdir(entities_data_path):
                entity_training_data = load_data(os.path.join(entities_data_path, entities_data))
                self.trainer.train(entity_training_data)
                self.trainer.persist(self.models_path, fixed_model_name=entities_data.split('.')[0])
        else:
            entity_training_data = load_data(os.path.join(entities_data_path, f'{fixed_intent_name}.json'))
            self.trainer.train(entity_training_data)
            self.trainer.persist(self.models_path, fixed_model_name=fixed_intent_name)

    """ Loads intent classification model """
    def load_intents_model(self):
        models_default_path = os.path.join(self.models_path, "default")
        self.intent_model = Interpreter.load(os.path.join(models_default_path, 'intents'), self.builder)

    """ Loads entity extraction model """
    def load_entities_model(self, fixed_model_name = None):
        models_default_path = os.path.join(self.models_path, "default")

        if fixed_model_name is None:
            models_available = os.listdir(models_default_path)

            for model in models_available:
                self.entity_models[model] = Interpreter.load(os.path.join(models_default_path, model), self.builder)
        else:
            self.entity_models[fixed_model_name] = Interpreter.load(os.path.join(models_default_path, fixed_model_name), self.builder)

    """ Parses text as intent object """
    def parse_as_intent(self, text):
        intent_name = self.__parse_text_intent(text)
        intent_data = self.__parse_text_entities(intent_name["intent"]["name"], text)
        
        return Intent(intent_data)

    def __parse_text_intent(self, text):
        intent = self.intent_model.parse(text)
        
        return intent

    def __parse_text_entities(self, fixed_intent_name, text):
        entities = self.entity_models[fixed_intent_name].parse(text)

        return entities
    
    """ Implementation of base class abstract method """
    def result_function(self, message: str) -> str:
        intent = self.parse_as_intent(message)

        result = dict()
        result["intent"] = intent.most_probable_intent["name"]
        result["entities"] = intent.entities
        result ["intent_ranking"] = intent.all_probable_intents

        return json.dumps(result)
    
    def initalise_nlu(self):
        self.load_intents_model()
        self.load_entities_model()
        self.loop()
    
    def train_all_models(self):
        self.train_intents_model()
        self.train_entitiy_models()

if __name__ == "__main__":
    nlu_config = {
    "config_path": os.environ["NLU_CONFIG"],
    "models_path": os.environ["NLU_MODELS"],
    "data_path" : os.environ["NLU_DATA"]
    }

    model = NLUModel(
        config_path=nlu_config["config_path"], 
        models_path=nlu_config["models_path"],
        data_path=nlu_config["data_path"]
    )
    
    methods = {
        "start": model.initalise_nlu,
        "train-models": model.train_all_models,
        "train-intents": model.train_intents_model,
        "train-entities": model.train_entitiy_models
    }

    try:
        if sys.argv[1].endswith(".py"):
            methods[sys.argv[2]]()
        else:
            methods[sys.argv[1]]()
    except IndexError:
        sys.stdout.write("Command not found")
        sys.stdout.flush()
    except KeyError:
        sys.stdout.write("Command not found")
        sys.stdout.flush()