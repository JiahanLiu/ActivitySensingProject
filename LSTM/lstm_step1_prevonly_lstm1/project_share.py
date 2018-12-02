import numpy as np
import os
from keras.callbacks import Callback
from extrasensory_lib import *
from pandas import DataFrame
from pandas import concat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
nobody needs to use this.
its mostly here to remind me of
useful functions
uuids -> the list of all uuids
num_tests -> the number of random uuids to choose

picks a random set of uuids from uuids. 
NOTE: it could pick the same uuid twice
might fix with np.unique 
"""
# def pick_test_uuids(uuids, num_tests):
#     picks = np.floor(np.random.rand(num_tests) * (len(uuids)+1))
#     picks = picks.astype(int)
#     np_uuids = np.array(uuids)
#     return uuids[picks]

"""
outputs the training loading from the provided prefix dir
ouputs the following tuple
(xtrn,ytrn,mtrn,xtst,ytst,mtst)
"""
def create_train_test_set(prefix_dir,uuids,tuuids):
    #x_trn,y_trn,m_trn
    #x_tst,y_tst,m_tst
    sensors_to_use = ['Acc','Gyro','WAcc','Loc','Aud','PS']

    num_tst = len(tuuids)
    num_trn = len(uuids) - num_tst
    #there are 225 features in total and 51 labels
    features = 175
    x_trn = np.empty((0,features))
    y_trn = np.empty((0,51))
    m_trn = np.empty((0,51))
    x_tst = np.empty((0,features))
    y_tst = np.empty((0,51))
    m_tst = np.empty((0,51))

    for uuid in uuids:
        (X,Y,M,q,feature_names,label_names) = read_user_data(prefix_dir + uuid);
        #print(feature_names)
        sense_feature_names = get_sensor_names_from_features(feature_names)
        #print("\n")
        #print(sense_feature_names)
        X=project_features_to_selected_sensors(X, sense_feature_names, sensors_to_use)
        
        if uuid  in tuuids:
            x_tst=np.vstack((x_tst,X));
            y_tst=np.vstack((y_tst,Y));
            m_tst=np.vstack((m_tst,M));
        else:
            x_trn=np.vstack((x_trn,X));
            y_trn=np.vstack((y_trn,Y));
            m_trn=np.vstack((m_trn,M));
    
    #standardize features: these issue runtime warnings but I think its okay
    (mean_vec,std_vec) = estimate_standardization_params(x_tst);
    x_tst = standardize_features(x_tst,mean_vec,std_vec);
    (mean_vec,std_vec) = estimate_standardization_params(x_trn);
    x_trn = standardize_features(x_trn,mean_vec,std_vec);
    #change NAN inputs to 0s (is okay because of standardization)
    x_tst[np.isnan(x_tst)] = 0.;
    x_trn[np.isnan(x_trn)] = 0.;
    
    #convert boolean arrays to integer arrays (not sure if dis really needed)
    y_trn = y_trn.astype(int)
    y_tst = y_tst.astype(int)
    m_trn = m_trn.astype(int)
    m_tst = m_tst.astype(int)
    
    return (x_trn,y_trn,m_trn,x_tst,y_tst,m_tst)

# class Project_Metrics(keras.callbacks.Callback):
class Project_Metrics(Callback):
    def __init__(self, test_mask):
        self.test_mask = test_mask
        
    def metric_calc(val, valmask, pre):
        m=valmask
        val = val.astype(bool)
        pre = pre.astype(bool)
        #total here isnt really accurate because missing shit
        total = val.shape[0]
        mtotal = m.sum(axis=0)
        #adjusted_total = total - mtotal
        mval = np.ma.masked_array(val, mask = m)
        mpre = np.ma.masked_array(pre, mask = m)
        
        Positives = mval.sum(axis=0)
        Negatives = total - mtotal - Positives
        
        TruePositives = (mval & mpre).sum(axis=0)
        TrueNegatives = (~(mval | mpre)).sum(axis=0)
        acc  = (TruePositives + TrueNegatives)/(Positives + Negatives)
        bacc = ((TruePositives/Positives)+(TrueNegatives/Negatives))/2
        sens = TruePositives/Positives
        spec = TrueNegatives/Negatives
        
        return (acc, sens, spec, bacc)
        
    def on_train_begin(self, logs={}):
        self._data = []
        self._detailed_data = []
        self.validation_data = [test_X,test_Y]

    def on_epoch_end(self, batch, logs={}):
        xval = self.validation_data[0]
        yval = self.validation_data[1]
        yval = yval.astype(int)
        ypre = np.asarray(self.model.predict(xval).round()).astype(int)
        (acc,sens,spec,bacc)=Project_Metrics.metric_calc(yval, self.test_mask,ypre)
        self._data.append([acc.mean(),sens.mean(),spec.mean(),bacc.mean()])
        self._detailed_data.append([acc,sens,spec,bacc])
        
    def get_data(self):
        return self._data
        
    def get_detailed_data(self):
        return self._detailed_data


def series_to_supervised(data, labels, n_in=1, n_out=1):
    n_vars = 1 if type(data) is list else data.shape[1]
    n_vars_label = 1 if type(labels) is list else labels.shape[1]
    df = DataFrame(data)
    df_label = DataFrame(labels)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df_label.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars_label)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars_label)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    return agg

def create_time_series(xtrn, ytrn, xtst, ytst):
    #combine for a universal set to follow machinelearningmastery tutorial
    trainSet = np.append(xtrn, ytrn, axis=1)
    testSet = np.append(xtst, ytst, axis=1)

    # convert to csv to check
    # xtrn_df = pd.DataFrame(xtrn[0:100])
    # ytrn_df = pd.DataFrame(ytrn[0:100])
    # train_df = pd.DataFrame(trainSet[0:100])
    # xtrn_df.to_csv("xtrn.csv")
    # ytrn_df.to_csv("ytrn.csv")
    # train_df.to_csv("train.csv")

    # data: x and y (all the data)
    # labels: y (just the labels)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_trainSet = scaler.fit_transform(trainSet)

    #creates a (271088, 277 complete time series)
    time_series_train = series_to_supervised(scaled_trainSet, ytrn, 1, 1);
    time_series_test = series_to_supervised(testSet, ytst, 1, 1);

    # first row doesnt have a t-1
    time_series_train = time_series_train.drop(time_series_train.index[0])
    time_series_test = time_series_test.drop(time_series_test.index[0])

    # print(xtrn.shape)
    # print(ytrn.shape)
    # print(scaled_trainSet.shape)
    # print(time_series_train.shape)

    # convert to csv to check
    # xtrn_df = pd.DataFrame(xtrn[0:100])
    # ytrn_df = pd.DataFrame(ytrn[0:100])
    # train_df = pd.DataFrame(scaled_trainSet[0:100])
    # time_series_train_df = time_series_train[0:100]
    # xtrn_df.to_csv("xtrn.csv")
    # ytrn_df.to_csv("ytrn.csv")
    # train_df.to_csv("train.csv")
    # time_series_train_df.to_csv("time_series_train.csv")

    return (time_series_train, time_series_test)

def reshape_time_series(time_series_train, time_series_test):
    time_series_train_np = time_series_train.values
    time_series_test_np = time_series_test.values

    train_X, train_Y = time_series_train_np[:, :-51], time_series_train_np[:,-51:]
    test_X, test_Y = time_series_test_np[:, :-51], time_series_test_np[:,-51:]

    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

    return (train_X, train_Y, test_X, test_Y)

#
# PROGRAM START 
#
data_dir = "../../extra_sensory_dataset/";
puuid_features_dir = "ExtraSensory.per_uuid_features_labels/";

#a common test set (size is 16 of 56)
test_uuids = ['0BFC35E2-4817-4865-BFA7-764742302A2D',
'1155FF54-63D3-4AB2-9863-8385D0BD0A13',
'CCAF77F0-FABB-4F2F-9E24-D56AD0C5A82F',
'0A986513-7828-4D53-AA1F-E02D6DF9561B',
'FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF',
'A5A30F76-581E-4757-97A2-957553A2C6AA',
'8023FE1A-D3B0-4E2C-A57A-9321B7FC755F',
'59EEFAE0-DEB0-4FFF-9250-54D2A03D0CF2',
'24E40C4C-A349-4F9F-93AB-01D00FB994AF',
'B09E373F-8A54-44C8-895B-0039390B859F',
'4FC32141-E888-4BFF-8804-12559A491D8C',
'61359772-D8D8-480D-B623-7C636EAD0C81',
'CDA3BBF7-6631-45E8-85BA-EEB416B32A3C',
'78A91A4E-4A51-4065-BDA7-94755F0BB3BB',
'481F4DD2-7689-43B9-A2AA-C8772227162B',
'806289BC-AD52-4CC1-806C-0CDB14D65EB6',
'BE3CA5A6-A561-4BBD-B7C9-5DF6805400FC',
'7D9BB102-A612-4E2A-8E22-3159752F55D8']

all_uuids = os.listdir(data_dir + puuid_features_dir);
#stupid filename nonesense fixing
for idx,str in enumerate(all_uuids):
    all_uuids[idx] = str[0:-23]

print("Begin data loading")
loaded_from_gz=False
need_time_series=False
persistent_filenames = ["xtrn_persistent","ytrn_persistent","mtrn_persistent","xtst_persistent","ytst_persistent","mtst_persistent"]
persistent_timeSeries = ["train_X", "train_Y", "test_X", "test_Y"]

for fname in persistent_filenames:
    if not os.path.isfile(fname+".npy"):
        loaded_from_gz = True
        break   
for fname in persistent_timeSeries:
    if not os.path.isfile(fname+".npy"):
        need_time_series = True
        break 

#if out of memory, do the training, and testing set seperately
if loaded_from_gz:
    print("loading from gx files")
    (xtrn,ytrn,mtrn,xtst,ytst,mtst) = create_train_test_set(data_dir + puuid_features_dir, all_uuids,test_uuids)
    np.save(persistent_filenames[0], xtrn)
    np.save(persistent_filenames[1], ytrn)
    np.save(persistent_filenames[2], mtrn)
    np.save(persistent_filenames[3], xtst)
    np.save(persistent_filenames[4], ytst)
    np.save(persistent_filenames[5], mtst)
else:
    print("loading from presaved files")
    if need_time_series:
        xtrn = np.load(persistent_filenames[0]+".npy")
        ytrn = np.load(persistent_filenames[1]+".npy")
        xtst = np.load(persistent_filenames[3]+".npy")
        ytst = np.load(persistent_filenames[4]+".npy")
    mtrn = np.load(persistent_filenames[2]+".npy")
    mtst = np.load(persistent_filenames[5]+".npy")

if need_time_series:
    print("doing time series")
    (time_series_train, time_series_test) = create_time_series(xtrn, ytrn, xtst, ytst)
    (train_X, train_Y, test_X, test_Y) = reshape_time_series(time_series_train, time_series_test)
    np.save(persistent_timeSeries[0], train_X)
    np.save(persistent_timeSeries[1], train_Y)
    np.save(persistent_timeSeries[2], test_X)
    np.save(persistent_timeSeries[3], test_Y)
else:
    print("loading time series")
    train_X = np.load(persistent_timeSeries[0]+".npy")
    train_Y = np.load(persistent_timeSeries[1]+".npy")
    test_X = np.load(persistent_timeSeries[2]+".npy")
    test_Y = np.load(persistent_timeSeries[3]+".npy")
    # print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

print("training/testing x/y shapes")
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

print("end data loading")