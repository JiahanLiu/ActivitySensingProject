import subprocess
import os

origWD = os.getcwd() # remember our original working directory

codeFilename = "generate_lstm.py";

subdirs = []

with open("masterList.txt") as f:
    content = f.readlines()

subdirs = [x.strip() for x in content] 


for dirs in subdirs:
	print(dirs)
	os.chdir(dirs)
	file = os.path.join(os.getcwd(), codeFilename)
	subprocess.call(["python", file])
	os.chdir(origWD)

