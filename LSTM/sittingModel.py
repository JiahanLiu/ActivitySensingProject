import pandas as pd
import os

#see which datafiles are available
from subprocess import check_output
datafilePath = "../data"; 
cmdAvailableDatafiles = check_output(["ls", "../data"]).decode("utf8")
availableDatafiles = cmdAvailableDatafiles.split('\n')
for index, datafile in enumerate(availableDatafiles):
	availableDatafiles[index] = os.path.join(datafilePath, datafile);

df = pd.read_csv(availableDatafiles[0])

#drop all labels except sitting
labelsToRemove = []
for label in list(df):
	if(label == 'label_source'):
		labelsToRemove.append(label)
	if((label[0:5] == 'label') & (label != 'label:SITTING')):
		labelsToRemove.append(label)
df = df.drop(labelsToRemove, axis=1)

