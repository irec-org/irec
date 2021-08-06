from googleDriveFileDownloader import googleDriveFileDownloader
from termcolor import colored
import zipfile
import shutil 
import os 

destination = "../../data/datasets/"
file_name = "all_train-tests.zip"

file = googleDriveFileDownloader()
file.downloadFile("https://drive.google.com/uc?id=1uKdXMnXdP8Np_9oGVyCvPFsqMaffdMDC&export=download", destination+file_name)

with zipfile.ZipFile(destination+file_name,"r") as zip_ref:
    zip_ref.extractall(destination)

for trte in os.listdir(destination+"TrainTest"):
	try:
		shutil.move(destination+"TrainTest/"+trte, destination) 
	except Exception as e:
		shutil.rmtree(destination+"TrainTest/"+trte)
		print(colored(e, "red"))

try:
	os.remove(destination+file_name)
	os.rmdir(destination+"TrainTest")
except Exception as e:
	print(colored(e, "red"))

print("Download completed:", destination+"...")