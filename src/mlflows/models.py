from trainer.trainers import ModelManager
from mlflows.registry import Registered_Model
from mlflows.registry import *

from pprint import pprint

if __name__ == '__main__':
    model_name = 'model'
    print(get_latest_model_versions('http://127.0.0.1:5000',model_name))
    transition_model_to_production('http://127.0.0.1:5000',model_name,'10')
    transition_model_to_staging('http://127.0.0.1:5000',model_name,'9')
