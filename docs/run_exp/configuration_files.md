# Configuration Files

iRec has some configuration files to define an experiment, such as dataset settings, agents, policies and evaluation metrics. Below we present brief examples about each of the files available in this framework.

For more details on configuration files, go to [**configuration_files**](https://github.com/irec-org/irec/tree/update-info/tutorials)

[**dataset_loaders.yaml**](https://github.com/irec-org/irec-cmdline/blob/main/app/settings/dataset_loaders.yaml)

This configuration file stores all the configurations related to the bases that will be used during the execution of an experiment.

```yaml
'MovieLens 100k': # Dataset name
    FullData: # Loader method
        dataset: # Info dataset
          path: ./data/datasets/MovieLens 100k/ratings.csv
          random_seed: 0
          file_delimiter: ","
          skip_head: true

        prefiltering: # Filters
          filter_users: # By Users
            min_consumption: 50
            num_users: 100
          filter_items: # By Items
            min_ratings: 1
            num_items: 100

        splitting: # Splitting
          strategy: temporal
          train_size: 0.8
          test_consumes: 5

        validation:
          validation_size: 0.2
︙
```

[**dataset_agents.yaml**](https://github.com/irec-org/irec-cmdline/blob/main/app/settings/dataset_agents.yaml)

This configuration file stores the settings of the agents (Recommendators) that will be used in the experiments.

```yaml
'MovieLens 100k':
  LinearUCB:
    SimpleAgent:
      action_selection_policy:
        ASPGreedy: {}
      value_function:
        LinearUCB:
          alpha: 1.0
          item_var: 0.01
          iterations: 20
          num_lat: 20
          stop_criteria: 0.0009
          user_var: 0.01
          var: 0.05
 ︙
```

[**agents_variables.yaml**](https://github.com/irec-org/irec-cmdline/blob/main/app/settings/agents_variables.yaml)

In this configuration file it is possible to define a search field for the variables of each agent, which will be used during the grid search

```yaml
GridSearch:

  - EGreedy:
      SimpleAgent:
        action_selection_policy:
          ASPEGreedy:
            epsilon: linspace(0.001, 1, 10)
        value_function:
          EGreedy: {}

  - UCB:
      SimpleAgent:
        action_selection_policy:
          ASPGreedy: {}
        value_function:
          UCB:
            c: linspace(0.001, 1, 10)
```

[**evaluation_policies.yaml**](app/settings/evaluation_policies.yaml)

In this configuration file the evaluation policies are defined. To carry out an experiment, we need to define how the recommendation process will be, the interactions between user and item, and for that we create an evaluation policy in accordance with the objectives of the experiment.

```yaml
FixedInteraction:
  num_interactions: 100
  interaction_size: 1
  save_info: False

︙
```

[**metric_evaluators.yaml**](app/settings/metric_evaluators.yaml)

This file defines the evaluation metrics for an experiment. This file is responsible for providing details on how to assess the interactions performed during the assessment process.

```yaml
UserCumulativeInteraction:
  interaction_size: 1
  interactions_to_evaluate:
    - 5
    - 10
    - 20
    - 50
    - 100
  num_interactions: 100
  relevance_evaluator_threshold: 3.999

︙
```

[**defaults.yaml**](app/settings/defaults.yaml)

This configuration file is a way to define the general settings for an experiment, here we can define the agents, the base, the policy and the evaluation metric, as well as some additional information.

```yaml
agent: LinearUCB
agent_experiment: agent
data_dir: data/
dataset_experiment: dataset
dataset_loader: 'MovieLens 1M'
evaluation_experiment: evaluation
evaluation_policy: FixedInteraction
metric: Hits
metric_evaluator: UserCumulativeInteraction
pdf_dir: pdf/
tex_dir: tex/
```

For more details, please take a look at our [tutorials](https://github.com/irec-org/irec/tree/master/tutorials)
