import numpy as np
from project_share import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pickle
from sklearn.utils import class_weight
from keras import backend as K

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# architect model
lstm_model = Sequential()
lstm_model.add(LSTM(60, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer='normal', activation='relu'))
lstm_model.add(Dense(51, kernel_initializer='normal', activation='sigmoid'))
# Compile model
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[sensitivity, specificity])

#need to adjust for missing rows
mtst_timeseries = mtst[1:]; 
bacc_metric = Project_Metrics(mtst_timeseries)
history = lstm_model.fit(train_X, train_Y, epochs=100, batch_size=10000, validation_data=(test_X, test_Y), callbacks=[bacc_metric])
print(bacc_metric.get_data())

lstm_model.save('lstm_step1_prevonly_lstm1.h5')  # creates a HDF5 file 'lstm_basic.h5'
with open('bacc_metric_simple.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(bacc_metric.get_data(), output)
with open('bacc_metric_detailed.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(bacc_metric.get_detailed_data(), output)