# Quick Start

Under app/ folder is a example of a application using irec and mlflow, where different experiments can be run with easy using existing recommender systems.

## Example

Check this example of a execution using the example application:

    cd app

    metrics=(Hits Precision Recall);
    models=(Random MostPopular UCB ThompsonSampling EGreedy);
    metric_evaluator="IterationsMetricEvaluator"
    evaluation_policy="Interaction"
    bases=("Netflix 10k" "Good Books" "Yahoo Music 10k");

    # run agents
    ./run_agent_best.py --dataset_loaders "${bases[@]}" --agents "${models[@]}" --evaluation_policy "$evaluation_policy"
  
    # evaluate agents using the metrics and metric evaluator defined
    ./eval_agent_best.py --dataset_loaders "${bases[@]}"\
    --agents "${models[@]}" --metrics "${metrics[@]}"\
    --evaluation_policy "$evaluation_policy"
    --metric_evaluator="$metric_evaluator"

    # print latex table with results and statistical test
    ./print_latex_table_results.py --dataset_loaders "${bases[@]}"\
    --agents "${models[@]}" --metrics "${metrics[@]}"\
    --evaluation_policy "$evaluation_policy"
    --metric_evaluator="$metric_evaluator"

For more details, please take a look at our [tutorials](run_example.ipynb)