from setuptools import setup, find_packages

setup(
    name='iRec',
    packages=find_packages(),
    url='https://github.com/irec-org/irec',
    description='Install iRec Framework',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy==1.20",
        "threadpoolctl>=3.0.0",
    	"tqdm>=4.62.3",
    	"gdown",
    	"mlflow",
    	"matplotlib",
    	"sklearn",
    	"numba",
    	"pyyaml",
        "traitlets",
        "behave",
        "behave_pandas",
        "cachetools",
        "tensorflow",
        ],
    include_package_data=True,
)