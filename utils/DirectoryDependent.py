from pathlib import Path
import os

class DirectoryDependent:
    BASE_DIR = ''
    DIRS = {}
    DIRS['data'] = os.path.join(BASE_DIR, 'data')
    DIRS['result'] = os.path.join(DIRS['data'], 'result')
    DIRS['state_save'] = os.path.join(DIRS['data'], 'state_save')
    DIRS['img'] = os.path.join(DIRS['data'], 'img')
    DIRS['export'] = os.path.join(DIRS['data'], 'export')
    DIRS['metric'] = os.path.join(DIRS['state_save'], 'metric')
    EXISTS = False
    def __init__(self):
        if not self.EXISTS:
            self.EXISTS = True
            for d in self.DIRS.values():
                Path(d).mkdir(parents=True, exist_ok=True)
