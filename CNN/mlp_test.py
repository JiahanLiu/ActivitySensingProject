import numpy as np
import pickle
from project_share import *
from tensorflow import keras

mlp_input_dim = 175
model_dir = "./models/"
metric_dir = "./metrics/"
#create models
mlp_2_16_16 = keras.Sequential()


mlp_2_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_16_16.add(keras.layers.Reshape((2,8),input_shape=(16,)))
mlp_2_16_16.add(keras.layers.Conv1D(16, 2, activation='relu', input_shape=(2,8)))
mlp_2_16_16.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

input_mlp_models = {
"mlp_2_16_16"           : mlp_2_16_16,
}

mlp_2_16_16_metrics         = Project_Metrics(mtst)

mlp_metrics = {
"mlp_2_16_16"           : mlp_2_16_16_metrics,
}

#compile models
for key in mlp_models:
    model = mlp_models[key]
    model.compile(loss='categorical_crossentropy', optimizer='adam')

# epochs : batch : est_time
# 100 : 10000 : 3h (6s/e)
# 100 : 1000  : 4h (8s/e)
# 100 : 100   : 13h(26s/e)

b = 100000
e = 3
for key in mlp_models:
    model = mlp_models[key]
    metric = mlp_metrics[key]
    
    print("training " + key)
    
    model.fit(xtrn,ytrn, epochs=e, batch_size=b, callbacks=[metric])
    mlp_metrics[key] = metric
    
    with open(metric_dir + key + '.avg.metric.pkl', 'wb') as output:
        pickle.dump(metric.get_data(), output, pickle.HIGHEST_PROTOCOL)
        
    with open(metric_dir + key + '.det.metric.pkl', 'wb') as output:
        pickle.dump(metric.get_detailed_data(), output, pickle.HIGHEST_PROTOCOL)
    
    model.save(model_dir + key + '.mdl')
