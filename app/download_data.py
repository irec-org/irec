#!/usr/bin/python3
import argparse
import zipfile
import shutil 
import gdown
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', nargs="*", required=True, type=str, help='Name of dataset')
args = parser.parse_args()

datasets_ids = {
	"Good Books": "14mT0bNCveFB1wKR_uhG08-GpOd_MxCZ7",
	"Good Reads 10k": "1j3JA8ZUhCAWruvKgSgY1LbYpc0Mdrig8",
	"Kindle 4k": "1CvhZ2GwalzHq9cp5r9SBKliedxfWpBba",
	"Kindle Store": "1JBCBBLDFcY46RKn8vw5u9EeJM_2-oASn",
	"LastFM 5k": "1AcnaOmxJccTaGuxAYH7icv9Vg_yh8r--",
	"MovieLens 1M": "1zQZ3vxEEXFIjpS8mS82B3XSw84SCPe6w",
	"MovieLens 10M": "1pV5PD2Cio41DLGEcN0Fb5MP1cGPzBJBB",
	"MovieLens 20M": "1LOStldkZgOKyaOjd8QhSs8dkM9rPo6dY",
	"MovieLens 100k": "1C0lHUQv73v58khSIKBE1VeVSldVsu_gE",
	"Netflix": "13H960S8-I2a-U3V_PmOC1TfOLZrkdm_h",
	"Netflix 10k": "1yynqrIW7GwTGXvuZ0ToPf-SEOGACSil3",
	"Yahoo Music": "1zWxmQ8zJvZQKBgUGK49_O6g6dQ7Mxhzn",
	"Yahoo Music 5k": "1c7HRu7Nlz-gbcc1-HsSTk98PYIZSQWEy",
	"Yahoo Music 10k": "1LMIMFjweVebeewn4D61KX72uIQJlW5d0"
}

dataset_dir = "./data/datasets/"
url = "https://drive.google.com/uc?id="

if not os.path.isdir(dataset_dir):
	os.makedirs(dataset_dir)

for dataset in args.dataset_name:
	print("\nDataset:", dataset)
	output = f"{dataset_dir}{dataset}.zip"
	gdown.download(f"{url}{datasets_ids[dataset]}", output)
	with zipfile.ZipFile(output,"r") as zip_ref:
	    zip_ref.extractall(dataset_dir)
	os.remove(output)
