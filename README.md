<p align="center">
  <img width=200 src="https://user-images.githubusercontent.com/28633659/138386006-1c0ba3d1-a6c1-4571-b111-92f2c90b8bb3.png">
</p>
<div align="center"><p>
    <a href="https://github.com/heitor57/irec/releases/latest">
      <img alt="Latest release" src="https://img.shields.io/github/v/release/heitor57/irec" />
    </a>
    <a href="https://github.com/heitor57/irec/pulse">
      <img alt="Last commit" src="https://img.shields.io/github/last-commit/heitor57/irec"/>
    </a>
    <a href="https://github.com/heitor57/irec/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/heitor57/irec?style=flat-square&logo=MIT&label=License" alt="License"
    />
    </a>
</p>

</div>

## Introduction

> For Python >= 3.6

Reinforcement Learning Recommender Systems Framework

Main features:

- Several state-of-the-art reinforcement learning models for the recommendation scenario
- Novelty, coverage and much more different type of online metrics
- Integration with the most used datasets for evaluating recommendation systems
- Flexible configuration
- Modular and reusable design
- Contains multiple evaluation policies currently used in the literature to evaluate reinforcement learning models
- Online Learning and Reinforcement Learning models
- Metrics and metrics evaluators are awesome to evaluate recommender systems in different ways

Also, we provide a amazing application created using the IRec library (under the app/ folder) that can be used to setup a experiment under 5~ minutes with parallel processes, log registry and results views. The main features are:

- Powerful application to run any reinforcement learning experiment powered by MLflow
- Entire pipeline of execution is fully parallelized
- Log registry
- Results views
- Statistical test
- Extensible environment


## Install

Install with pip:

    pip install irec


## Examples

Under app/ folder is a example of a application using irec and mlflow, where different experiments can be run with easy using existing recommender systems.

Check this example of a execution using the example application:

    cd app
    metrics=(Hits Precision Recall);
    models=(Random MostPopular UCB ThompsonSampling EGreedy);
    metric_evaluator="IterationsMetricEvaluator"
    bases=("Netflix 10k" "Good Books" "Yahoo Music 10k");
    # run agents
    ./run_agent_best.py --dataset_loaders "${bases[@]}" --agents "${models[@]}"
  
    # evaluate agents using the metrics and metric evaluator defined
    ./eval_agent_best.py --dataset_loaders "${bases[@]}"\
    --agents "${models[@]}" --metrics "${metrics[@]}"\
    --metric_evaluator="$metric_evaluator"

    # print latex table with results and statistical test
    ./print_latex_table_results.py --dataset_loaders "${bases[@]}"\
    --agents "${models[@]}" --metrics "${metrics[@]}"\
    --metric_evaluator="$metric_evaluator"

Also, using the framework is also awesome, check these examples:

    TODO

## API

For writing anything new to the library (e.g., value function, agent, etc) read the documentation.

## Contributing

All contributions are welcome! Just open a pull request.
