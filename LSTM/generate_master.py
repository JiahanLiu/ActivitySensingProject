import os

allFilesInDir = os.listdir("./")
subdirs = []
for dirs in allFilesInDir:
	if os.path.isdir(dirs):
		subdirs.append(dirs)

with open("masterList.txt", "w") as text_file:
	for dirs in subdirs:
		text_file.writelines(dirs)
		text_file.writelines("\n")