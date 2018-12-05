import pickle
import os
import matplotlib.pyplot as plt

allFilesInDir = os.listdir("./")
subdirs = []
for dirs in allFilesInDir:
	if os.path.isdir(dirs):
		subdirs.append(dirs)

codeFilename = "bacc_metric_simple.pkl";

with open(os.path.join('lstm_prediction_stacked_thick_dropout_3/', codeFilename), 'rb') as input:  # Overwrites any existing file.
	metric_simple = pickle.load(input)
	i = 1
	single_run_x = []
	single_run_y = []
	for x in metric_simple:
		single_run_x.append(i)
		single_run_y.append(x[3])
		i = i + 1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x=single_run_x, y=single_run_y, label='linear')
fig.savefig('test.png')
