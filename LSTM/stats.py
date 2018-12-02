from keras.models import load_model
from keras.callbacks import Callback
import numpy as np
import pickle

# model = load_model('lstm_step1_prevonly_lstm1.h5')

with open('bacc_metric_simple.pkl', 'rb') as input:  # Overwrites any existing file.
	metric_simple = pickle.load(input)
with open('bacc_metric_detailed.pkl', 'rb') as input:  # Overwrites any existing file.
	metric_detailed = pickle.load(input)

print(metric_simple)
print(metric_detailed)