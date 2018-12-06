import numpy as np
import pickle
from project_share import *
from tensorflow import keras

mlp_input_dim = 175
model_dir = "./models/"
metric_dir = "./metrics/"
#create models
#linear = keras.Sequential()
#mlp_1_32 = keras.Sequential()
#mlp_1_256 = keras.Sequential()
mlp_2_16_16 = keras.Sequential()
mlp_2_8_16 = keras.Sequential()
mlp_2_16_8 = keras.Sequential()
mlp_2_16_32 = keras.Sequential()
mlp_2_32_16 = keras.Sequential()
mlp_3_8_8_8 = keras.Sequential()
mlp_3_16_16_16 = keras.Sequential()
mlp_3_32_32_32 = keras.Sequential()
mlp_3_8_16_8 = keras.Sequential()
mlp_3_32_16_32 = keras.Sequential()
mlp_3_16_32_16 = keras.Sequential()
mlp_3_32_64_32 = keras.Sequential()
mlp_5_16_16_16_16_16 = keras.Sequential()
mlp_5_16_32_64_32_16 = keras.Sequential()
mlp_5_64_32_16_32_64 = keras.Sequential()

#linear.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

#mlp_1_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
#mlp_1_32.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

#mlp_1_256.add(keras.layers.Dense(256, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
#mlp_1_256.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_2_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_16_16.add(keras.layers.Reshape((2,8),input_shape=(16,)))
mlp_2_16_16.add(keras.layers.Conv1d(16, 2, activation='relu', input_shape=(2,8)))
mlp_2_16_16.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_2_8_16.add(keras.layers.Dense(8, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_8_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_8_16.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_2_16_8.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_16_8.add(keras.layers.Dense(8, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_16_8.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_2_16_32.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_16_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_16_32.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_2_32_16.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_32_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_32_16.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_3_8_8_8.add(keras.layers.Dense(8, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_8_8_8.add(keras.layers.Dense(8, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_8_8_8.add(keras.layers.Dense(8, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_8_8_8.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_3_16_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_16_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_16_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_16_16_16.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_3_32_32_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_32_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_32_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_32_32.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_3_8_16_8.add(keras.layers.Dense(8, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_8_16_8.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_8_16_8.add(keras.layers.Dense(8, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_8_16_8.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_3_32_16_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_16_32.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_16_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_16_32.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_3_16_32_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_16_32_16.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_16_32_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_16_32_16.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_3_32_64_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_64_32.add(keras.layers.Dense(64, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_64_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_64_32.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_5_16_16_16_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_16_16_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_16_16_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_16_16_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_16_16_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_16_16_16_16.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_5_16_32_64_32_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_32_64_32_16.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_32_64_32_16.add(keras.layers.Dense(64, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_32_64_32_16.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_32_64_32_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_32_64_32_16.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_5_64_32_16_32_64.add(keras.layers.Dense(64, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_64_32_16_32_64.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_64_32_16_32_64.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_64_32_16_32_64.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_64_32_16_32_64.add(keras.layers.Dense(64, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_64_32_16_32_64.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_models = {
#"linear"                : linear,
#"mlp_1_32"              : mlp_1_32,
#"mlp_1_256"             : mlp_1_256,
"mlp_2_16_16"           : mlp_2_16_16,
"mlp_2_8_16"            : mlp_2_8_16,
"mlp_2_16_8"            : mlp_2_16_8,
"mlp_2_16_32"           : mlp_2_16_32,
"mlp_2_32_16"           : mlp_2_32_16,
"mlp_3_8_8_8"           : mlp_3_8_8_8,
"mlp_3_16_16_16"        : mlp_3_16_16_16,
"mlp_3_32_32_32"        : mlp_3_32_32_32,
"mlp_3_8_16_8"          : mlp_3_8_16_8,
"mlp_3_32_16_32"        : mlp_3_32_16_32,
"mlp_3_16_32_16"        : mlp_3_16_32_16,
"mlp_3_32_64_32"        : mlp_3_32_64_32,
"mlp_5_16_16_16_16_16"  : mlp_5_16_16_16_16_16,
"mlp_5_16_32_64_32_16"  : mlp_5_16_32_64_32_16,
"mlp_5_64_32_16_32_64"  : mlp_5_64_32_16_32_64
}

linear_metrics              = Project_Metrics(mtst)
mlp_1_32_metrics            = Project_Metrics(mtst)
mlp_1_256_metrics           = Project_Metrics(mtst)
mlp_2_16_16_metrics         = Project_Metrics(mtst)
mlp_2_8_16_metrics          = Project_Metrics(mtst)
mlp_2_16_8_metrics          = Project_Metrics(mtst)
mlp_2_16_32_metrics         = Project_Metrics(mtst)
mlp_2_32_16_metrics         = Project_Metrics(mtst)
mlp_3_8_8_8_metrics         = Project_Metrics(mtst)
mlp_3_16_16_16_metrics      = Project_Metrics(mtst)
mlp_3_32_32_32_metrics      = Project_Metrics(mtst)
mlp_3_8_16_8_metrics        = Project_Metrics(mtst)
mlp_3_32_16_32_metrics      = Project_Metrics(mtst)
mlp_3_16_32_16_metrics      = Project_Metrics(mtst)
mlp_3_32_64_32_metrics      = Project_Metrics(mtst)
mlp_5_16_16_16_16_16_metrics= Project_Metrics(mtst)
mlp_5_16_32_64_32_16_metrics= Project_Metrics(mtst)
mlp_5_64_32_16_32_64_metrics= Project_Metrics(mtst)

mlp_metrics = {
#"linear"                : linear_metrics,
#"mlp_1_32"              : mlp_1_32_metrics,
#"mlp_1_256"             : mlp_1_256_metrics,
"mlp_2_16_16"           : mlp_2_16_16_metrics,
"mlp_2_8_16"            : mlp_2_8_16_metrics,
"mlp_2_16_8"            : mlp_2_16_8_metrics,
"mlp_2_16_32"           : mlp_2_16_32_metrics,
"mlp_2_32_16"           : mlp_2_32_16_metrics,
"mlp_3_8_8_8"           : mlp_3_8_8_8_metrics,
"mlp_3_16_16_16"        : mlp_3_16_16_16_metrics,
"mlp_3_32_32_32"        : mlp_3_32_32_32_metrics,
"mlp_3_8_16_8"          : mlp_3_8_16_8_metrics,
"mlp_3_32_16_32"        : mlp_3_32_16_32_metrics,
"mlp_3_16_32_16"        : mlp_3_16_32_16_metrics,
"mlp_3_32_64_32"        : mlp_3_32_64_32_metrics,
"mlp_5_16_16_16_16_16"  : mlp_5_16_16_16_16_16_metrics,
"mlp_5_16_32_64_32_16"  : mlp_5_16_32_64_32_16_metrics,
"mlp_5_64_32_16_32_64"  : mlp_5_64_32_16_32_64_metrics
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
