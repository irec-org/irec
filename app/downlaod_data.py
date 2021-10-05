#!/usr/bin/python3
from googleDriveFileDownloader import googleDriveFileDownloader
from termcolor import colored
import zipfile
import shutil 
import os 

destination = "./data/datasets/"
file_name = "datasets.zip"

file = googleDriveFileDownloader()

# tr-te
# file.downloadFile("https://drive.google.com/uc?id=1uKdXMnXdP8Np_9oGVyCvPFsqMaffdMDC&export=download", destination+file_name)
# dirr = "TrainTest"

# datasets
file.downloadFile("https://drive.google.com/uc?id=1zy2k0jz3t4oA59w9cwMbNS7vFx_8ky3I&export=download", destination+file_name)
dirr = "New Datasets/"

with zipfile.ZipFile(destination+file_name,"r") as zip_ref:
    zip_ref.extractall(destination)

for trte in os.listdir(destination+dirr):
	try:
		shutil.move(destination+dirr+trte, destination) 
	except Exception as e:
		shutil.rmtree(destination+dirr+trte)
		print(colored(e, "red"))

try:
	os.remove(destination+file_name)
	os.rmdir(destination+dirr)
except Exception as e:
	print(colored(e, "red"))

print("Download completed:", destination+"...")
