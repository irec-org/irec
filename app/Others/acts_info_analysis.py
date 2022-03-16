from os.path import dirname, realpath, sep, pardir
import sys
import os
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

from mlflow.tracking import MlflowClient
from irec.utils.dataset import Dataset
import scipy.sparse
import irec.metrics
import pandas as pd
import numpy as np
import irec.agents
import inquirer
import argparse
import irec.mf
import mlflow
import metrics
import pickle
import ctypes
import utils
import yaml

settings = utils.load_settings()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_loaders", nargs="*", default=[settings["defaults"]["dataset_loader"]])
parser.add_argument("--agents", nargs="*", default=[settings["defaults"]["agent"]])

utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

dataset_agents = yaml.load(open("./settings/dataset_agents.yaml"), Loader=yaml.SafeLoader)
mlflow.set_experiment(settings["defaults"]["agent_experiment"])

for dataset_loader_name in args.dataset_loaders:

    settings["defaults"]["dataset_loader"] = dataset_loader_name
    traintest_dataset = utils.load_dataset_experiment(settings)
    data = np.vstack((traintest_dataset.train.data, traintest_dataset.test.data))
    dataset = Dataset(data)
    dataset.update_from_data()
    dataset.update_num_total_users_items()

    consumption_matrix = scipy.sparse.csr_matrix(
        (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
        (dataset.num_total_users, dataset.num_total_items))

    evaluation_policy_name = settings["defaults"]["evaluation_policy"]
    evaluation_policy_parameters = settings["evaluation_policies"][evaluation_policy_name]
    evaluation_policy = eval("irec.evaluation_policies." + evaluation_policy_name)(**evaluation_policy_parameters)

    for agent_name in args.agents:

        settings["defaults"]["dataset_loader"] = dataset_loader_name
        settings["defaults"]["agent"] = agent_name
        agent_parameters = dataset_agents[dataset_loader_name][agent_name]
        settings["agents"][agent_name] = agent_parameters
        agent = utils.create_agent(agent_name, agent_parameters)
        agent_id = utils.get_agent_id(agent_name, agent_parameters)
        dataset_parameters = settings["dataset_loaders"][dataset_loader_name]

        run = utils.get_agent_run(settings)
        client = MlflowClient()
        
        artifact_path = client.download_artifacts(run.info.run_id, "interactions.pickle")
        users_items_recommended = pickle.load(open(artifact_path, "rb"))

        artifact_path = client.download_artifacts(run.info.run_id, "acts_info.pickle")
        acts_info = pickle.load(open(artifact_path, "rb"))

        data = []
        for i in range(len(acts_info)):
            uid = users_items_recommended[i][0]
            iid = users_items_recommended[i][1]
            if isinstance(agent,irec.agents.SimpleEnsembleAgent):
                data.append({
                    **acts_info[i],
                    **{
                        'uid': uid,
                        'iid': iid,
                        'reward': consumption_matrix[uid, iid]
                    }
                })
            else:
                data.append(
                        [uid,
                        iid,
                        consumption_matrix[uid, iid],]
                )

        if isinstance(agent,irec.agents.SimpleEnsembleAgent):
            df_results = pd.DataFrame(data)
            results = df_results.groupby(['user_interaction', 'meta_action_name'])['trial'].agg(['count'])
            print(results)
            results.to_csv(f'export/{dataset_loader_name}_{agent_name}_{agent_methods}.csv')
            print(df_results.groupby('meta_action_name')['reward'].mean())
        
        df_all_results=pd.DataFrame(data,columns=['uid','iid','reward'])
        df_all_results.to_parquet(open(f'export/d_{dataset_loader_name}_{agent_name}.parquet','wb'))
