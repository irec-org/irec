from typing import Dict
from irec.environment.dataset import Dataset
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from irec.connector.utils import run_agent
import copy

class Tunning:
    """docstring for Tunning"""

    def __init__(self):
        pass

    def generate_settings(self) -> Dict:
        pass

    def execute(self, x_validation:Dataset, y_validation:Dataset, agents_search_parameters, settings, tasks, forced_run):
        
        with ProcessPoolExecutor(max_workers=tasks) as executor:
            futures = set()
            for agent_name in agents_search_parameters:
                settings["defaults"]["agent"] = agent_name
                for agent_og_parameters in agents_search_parameters[agent_name]:
                    settings["agents"][agent_name] = agent_og_parameters
                    f = executor.submit(
                        # run_agent, train_dataset, test_dataset, copy.deepcopy(settings), forced_run
                        run_agent, x_validation, y_validation, copy.deepcopy(settings), forced_run
                    )
                    futures.add(f)
                    if len(futures) >= tasks:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in futures:
                f.result()
