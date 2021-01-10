# import sys
# sys.path.append('../lib')
import utils.dataset as utils
import yaml

fpath = "./settings/datasets.yaml"
loader = yaml.SafeLoader
datasets_settings = yaml.load(fpath,Loader=loader)
print(datasets_settings)
# dsf = DatasetFormatter()
# dsf.gen_base()
