# Quick Start

Under app/ folder is a example of a application using irec and mlflow, where different experiments can be run with easy using existing recommender systems.

## Example

Check this example of a execution using the example application:

    dataset=("Netflix 10k" "Good Books" "Yahoo Music 10k");\
    models=(Random MostPopular UCB ThompsonSampling EGreedy);\
    metrics=(Hits Precision Recall);\
    eval_pol=("FixedInteraction");
    metric_evaluator="Interaction";\
    
    cd agents &&
    python run_agent_best.py --agents "${models[@]}" --dataset_loaders "${dataset[@]}" --evaluation_policy "${eval_pol[@]}" &&
  
    cd ../evaluation &&
    python eval_agent_best.py --agents "${models[@]}" --dataset_loaders "${dataset[@]}" --evaluation_policy "${eval_pol[@]}" --metrics "${metrics[@]}" --metric_evaluator "${metric_eval[@]}" &&

    python print_latex_table_results.py --agents "${models[@]}" --dataset_loaders "${dataset[@]}" --evaluation_policy "${eval_pol[@]}" --metric_evaluator "${metric_eval[@]}" --metrics "${metrics[@]}"

For more details, please take a look at our [tutorials](https://github.com/irec-org/irec/tree/update-info/tutorials)
