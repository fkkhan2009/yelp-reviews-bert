import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd
import sys
# clone the tensorflow models garden
sys.path.append('models') # this step is required to use the next step
from official.nlp.data import classifier_data_lib #models/official
from official.nlp.bert import tokenization
from official.nlp import optimization