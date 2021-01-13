from pathlib import Path
import os
from os import sep, pardir

class DirectoryDependent:
    BASE_DIR = (pardir+sep)*2
    DIRS = {}
    DIRS['data'] = os.path.join(BASE_DIR, 'data')
    DIRS['results'] = os.path.join(DIRS['data'], 'results')
    DIRS['state_save'] = os.path.join(DIRS['data'], 'state_save')
    DIRS['dataset_preprocess'] = os.path.join(DIRS['data'], 'dataset_preprocess')
    DIRS['img'] = os.path.join(DIRS['data'], 'img')
    DIRS['export'] = os.path.join(DIRS['data'], 'export')
    DIRS['metric'] = os.path.join(DIRS['state_save'], 'metric')
    DIRS['datasets'] = os.path.join(DIRS['data'], 'datasets')
    EXISTS = False
    def __init__(self):
        if not self.EXISTS:
            self.EXISTS = True
            for d in self.DIRS.values():
                Path(d).mkdir(parents=True, exist_ok=True)
