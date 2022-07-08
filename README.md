<p align="center">
  <img width=200 src="https://user-images.githubusercontent.com/28633659/143366444-05c63b55-00f8-4221-8b30-e8d660f10bea.png">
</p>

<div align="center"><p>
    <a href="https://github.com/irec-org/irec/releases/latest">
      <img alt="Latest release" src="https://img.shields.io/github/v/release/irec-org/irec" />
    </a>
    <a href="https://github.com/irec-org/irec/pulse">
      <img alt="Last commit" src="https://img.shields.io/github/last-commit/irec-org/irec"/>
    </a>
    <a href="https://github.com/irec-org/irec/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/irec-org/irec?style=flat-square&logo=MIT&label=License" alt="License"
    />
    </a>
    <a href="https://github.com/irec-org/irec/actions/workflows/tests.yml">
      <img src="https://github.com/irec-org/irec/actions/workflows/tests.yml/badge.svg" alt="Tests"
    />
    </a>
    <a href="https://badge.fury.io/py/irec"><img src="https://badge.fury.io/py/irec.svg" alt="PyPI version" height="18"></a>
</p>

</div>

<div align="center">
	<a href="https://irec-org.github.io/irec/">Introduction</a>
  <span> • </span>
    	<a href="https://irec-org.github.io/irec/guide/install_irec/">Install</a>
  <span> • </span>
       	<a href="https://irec-org.github.io/irec/guide/quick_start/">Quick Start</a>
  <p></p>
</div>

![Overview iRec](./figures/IREC3.jpg)

iRec is structured in three main components as usually made by classical frameworks in the RS field. These main components are:

• **Environment Setting**: responsible for loading, preprocessing, and splitting the dataset into train and test sets (when required) to create the task environment for the pipeline;

• **Recommendation Agent**: responsible for implementing the recommendation model required as an interactive algorithm that will interact with the environment;

• **Experimental Evaluation**: responsible for defining how the agent will interact with the environment to simulate the interactive scenario and get the logs required for a complete evaluation.

## Introduction

> For Python >= 3.8

Interactive Recommender Systems Framework

Main features:

- Several state-of-the-art reinforcement learning models for the recommendation scenario
- Novelty, coverage and much more different type of online metrics
- Integration with the most used datasets for evaluating recommendation systems
- Flexible configuration
- Modular and reusable design
- Contains multiple evaluation policies currently used in the literature to evaluate reinforcement learning models
- Online Learning and Reinforcement Learning models
- Metrics and metrics evaluators are awesome to evaluate recommender systems in different ways

Also, we provide a amazing application created using the iRec library, the [iRec-cmdline](https://github.com/irec-org/irec-cmdline), that can be used to setup a experiment under 5~ minutes with parallel processes, log registry and results views. The main features are:

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

[iRec-cmdline](https://github.com/irec-org/irec-cmdline) contains a example of a application using iRec and MLflow, where different experiments can be run with easy using existing recommender systems.

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

## Datasets

Our framework has the ability to use any type of dataset, as long as it is suitable for the recommendation domain and is formatted correctly. Below we list some datasets tested and used in some of our experiments.

| Dataset | Domain  | Sparsity | Link
| :---: | --- | :---: | :---: |
MovieLens 100k |  Movies | 93.69% | [Link](https://grouplens.org/datasets/movielens/100k/)
MovieLens 1M | Movies | 95.80% | [Link](https://grouplens.org/datasets/movielens/1m/)
MovieLens 10M |  Movies | 98.66% | [Link](https://grouplens.org/datasets/movielens/10m/)
Netflix | Movies | 98.69% | [Link](link)
Ciao DVD | Movies | 99.97% | [Link](http://konect.cc/networks/librec-ciaodvd-movie_ratings/)	
Yahoo Music |  Musics | 97.63% | [Link](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r)
LastFM | Musics | 99.84% | [Link](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html)
Good Books |  Books | 98.88% | [Link](https://www.kaggle.com/zygmunt/goodbooks-10k?select=ratings.csv)
Good Reads | Books | 99.50% | [Link](https://www.kaggle.com/sahilkirpekar/goodreads10k-dataset-cleaned?select=Ratings.csv)
Amazon Kindle Store | Products | 99.97% | [Link](https://jmcauley.ucsd.edu/data/amazon/)	
Clothing Fit | Clothes | 99.97% | [Link](link)	

## Models

The [recommender models](https://github.com/irec-org/irec/tree/master/irec/recommendation/agents/value_functions) supported by irec are listed below.

| Year | Model  | Paper | Description
| :---: | --- | :---: | :--- |
| 2002 | [ε-Greedy](irec/recommendation/agents/value_functions/e_greedy.py) | [Link](https://link.springer.com/article/10.1023/A:1013689704352) | In general, ε-Greedy models the problem based on an ε diversification parameter to perform random actions.   
| 2013 | [Linear ε-Greedy](irec/recommendation/agents/value_functions/linear_egreedy.py) | [Link](https://dl.acm.org/doi/abs/10.1145/2505515.2505690?casa_token=1PDIAs6p1ysAAAAA:ZFkzkEnCX1_ZiqSCAgqOw9Z3mOPybhJLRtAdkfnEagDI_aef1TR7SD3IZkkVhs2hTzk_FkigZ548) | A linear exploitation of the items latent factors defined by a PMF formulation that also explore random items with probability ε.   
| 2011 | [Thompson Sampling](irec/recommendation/agents/value_functions/thompson_sampling.py) | [Link](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.831.5818&rep=rep1&type=pdf) |  A basic item-oriented bandit algorithm that follows a Gaussian distribution of items and users to perform the prediction rule based on their samples.
| 2013 | [GLM-UCB](irec/recommendation/agents/value_functions/glm_ucb.py) | [Link](https://dl.acm.org/doi/abs/10.1145/2505515.2505690?casa_token=cCSF9jXF2VMAAAAA:zpD_LhYXadYz5BAm5-R_SpSA4za8EGH4U98mbbxquS6BZFLFM2tylXVemkgW9033knXcqB_kumP5) | It follows a similar process as Linear UCB based on the PMF formulation, but it also adds a sigmoid form in the exploitation step and makes a time-dependent exploration.   
| 2018 | [ICTR](irec/recommendation/agents/value_functions/ictr.py) | [Link](https://ieeexplore.ieee.org/abstract/document/8440090/) | It is an interactive collaborative topic regression model that utilizes the TS bandit algorithm and controls the items dependency by a particle learning strategy.
| 2015 | [PTS](irec/recommendation/agents/value_functions/pts.py) | [Link](http://papers.nips.cc/paper/5985-efficient-thompson-sampling-for-online-matrix--factorization-recommendation.pdf) | It is a PMF formulation for the original TS based on a Bayesian inference around the items. This method also applies particle filtering to guide the exploration of items over time.   
| 2019 | [kNN Bandit](irec/recommendation/agents/value_functions/knn_bandit.py) | [Link](https://dl.acm.org/doi/abs/10.1145/3298689.3347040) | A simple multi-armed bandit elaboration of neighbor-based collaborative filtering. A variant of the nearest-neighbors scheme, but endowed with a controlled stochastic exploration capability of the users’ neighborhood, by a parameter-free application of Thompson sampling. 
| 2017 | [Linear TS](irec/recommendation/agents/value_functions/linear_ts.py) | [Link](http://proceedings.mlr.press/v54/abeille17a) | An adaptation of the original Thompson Sampling to measure the latent dimensions by a PMF formulation.
| 2013 | [Linear UCB](irec/recommendation/agents/value_functions/linear_ucb.py) | [Link](https://dl.acm.org/doi/abs/10.1145/2505515.2505690?casa_token=qJlau402jGcAAAAA:Wo57x-INaZl6bhdv-V8ab391I2GJ6SWxE9unEnRtg0urLVS5sISCdhR0REJRJGlM9bdAb3Jw_Fti) | An adaptation of the original LinUCB (Lihong Li et al. 2010) to measure the latent dimensions by a PMF formulation.   
| 2020 | [NICF](irec/recommendation/agents/value_functions/nicf.py) | [Link](https://dl.acm.org/doi/abs/10.1145/2505515.2505690?casa_token=MllrAXlioLsAAAAA:qnXgeSAEJF1jhTD7PNiDWFFr-FAET4vOHluesRPSCvxGuw3EfEeSnYokqCQKj3cNH0-v_I43UQE0) | It is an interactive method based on a combination of neural networks and  collaborative filtering that also performs a meta-learning of the user’s preferences.   
| 2016 | [COFIBA](irec/recommendation/agents/value_functions/cofiba.py) | [Link](https://dl.acm.org/doi/abs/10.1145/2911451.2911548?casa_token=UpXzuWNaGHUAAAAA:jQR2gPPq2plKCg2mqLMoJAn5l6BBd2fWi4oxw9DJN0LZ9r-03PLqb8qEKuNDD0DXcgp6N8W6x39b) | This method relies on upper-confidence-based tradeoffs between exploration and exploitation, combined with adaptive clustering procedures at both the user and the item sides.
| 2002 | [UCB](irec/recommendation/agents/value_functions/ucb.py) | [Link](https://link.springer.com/article/10.1023/A:1013689704352) | It is the original UCB that calculates a confidence interval for each item at each iteration and tries to shrink the confidence bounds.
| 2021 | [Cluster-Bandit (CB)](irec/recommendation/agents/value_functions/cluster_bandit.py) | [Link](https://dl.acm.org/doi/abs/10.1145/3404835.3463033) | it is a new bandit algorithm based on clusters to face the cold-start problem.
| 2002 | [Entropy](irec/recommendation/agents/value_functions/entropy.py) | [Link](https://dl.acm.org/doi/pdf/10.1145/502716.502737?casa_token=tQ6DkQMJnW0AAAAA:d3kGkV18mjoXwEDDMQmy4UBRMe9ZoZ-mCOeOqkZKgVCiIRpGolKB2M0RXm4ouePTuWkOgVhgBKh7) | The entropy of an item i is calculated using the relative frequency of the possible ratings. In general, since entropy measures the spread of ratings for an item, this strategy tends to promote rarely rated items, which can be considerably informative.
| 2002 | [log(pop)*ent](irec/recommendation/agents/value_functions/log_pop_ent.py) | [Link](https://dl.acm.org/doi/pdf/10.1145/502716.502737?casa_token=tQ6DkQMJnW0AAAAA:d3kGkV18mjoXwEDDMQmy4UBRMe9ZoZ-mCOeOqkZKgVCiIRpGolKB2M0RXm4ouePTuWkOgVhgBKh7) | It combines popularity and entropy to identify potentially relevant items that also have the ability to add more knowledge to the system. As these concepts are not strongly correlated, it is possible to achieve this combination through a linear combination of the popularity ρ of an item i by its entropy ε: score(i) = log(ρi) · εi.
| - | [Random](irec/recommendation/agents/value_functions/random.py) | [Link](link) | This method recommends totally random items.  
| - | [Most Popular](irec/recommendation/agents/value_functions/most_popular.py) | [Link](link) | It recommends items with the higher number of ratings received (most-popular) at each iteration.  
| - | [Best Rated](irec/recommendation/agents/value_functions/best_rated.py) | [Link](link) | Recommends top-rated items based on their average ratings in each iteration.

<!-- | 2021 | [WSPB](reposit) | [Link](link) | The best -->
<!-- | 2013 | [LinEgreedy](reposit) | [Link](https://dl.acm.org/doi/abs/10.1145/2505515.2505690?casa_token=qJlau402jGcAAAAA:Wo57x-INaZl6bhdv-V8ab391I2GJ6SWxE9unEnRtg0urLVS5sISCdhR0REJRJGlM9bdAb3Jw_Fti) | -->
<!-- | 2010 | [LinUCB](reposit) | [Link](https://dl.acm.org/doi/abs/10.1145/1772690.1772758?casa_token=0DH2lK1XlgMAAAAA:rw-99PWUUWSR5sH7lfOs3bcn_2wahVraUPE7l7iqGh8p3d2mFBuvYnKax-HirKgEMqGTrCjceJMv) | adaptation of LinUCB, defining the contexts as latent factors of the SVD and exploring the uncertainty of the items and users through the confidence interval ||qi||. -->

## Metrics

The [recommender metrics](https://github.com/irec-org/irec/tree/master/irec/offline_experiments/metrics) supported by iRec are listed below.

| Metric | Reference | Description
| :---: | --- | :--- |
| [Hits](irec/offline_experiments/metrics/hits.py) | [Link](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1) | Number of recommendations made successfully. 
| [Precision](irec/offline_experiments/metrics/precision.py) | [Link](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1) | Precision is defined as the percentage of predictions we get right.
| [Recall](irec/offline_experiments/metrics/recall.py) | [Link](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1) | Represents the probability that a relevant item will be selected.  
| [EPC](irec/offline_experiments/metrics/epc.py) | [Link](https://dl.acm.org/doi/abs/10.1145/2043932.2043955?casa_token=-c17w4Nyg4AAAAAA:olXeR-HjoDJ-CTnyJ5DE7uhM5LChpozaO73W1T8oIAnVqPv_fJndR99lhguMVTEnRl8SdqujvIdT3ok) | Represents the novelty for each user and it is measured by the expected number of seen relevant recommended items not previously seen.  
| [EPD](irec/offline_experiments/metrics/epd.py) | [Link](https://dl.acm.org/doi/abs/10.1145/2043932.2043955?casa_token=-c17w4Nyg4AAAAAA:olXeR-HjoDJ-CTnyJ5DE7uhM5LChpozaO73W1T8oIAnVqPv_fJndR99lhguMVTEnRl8SdqujvIdT3ok) | EPD is a distance-based novelty measure, which looks at distances between the items in the user’s profile and the recommended items. 
| [ILD](irec/offline_experiments/metrics/ild.py) | [Link](https://dl.acm.org/doi/abs/10.1145/2043932.2043955?casa_token=-c17w4Nyg4AAAAAA:olXeR-HjoDJ-CTnyJ5DE7uhM5LChpozaO73W1T8oIAnVqPv_fJndR99lhguMVTEnRl8SdqujvIdT3ok) | It represents the diversity between the list of items recommended. This diversity is measured by the Pearson correlation of the item’s features vector. 
| [Gini Coefficient](irec/offline_experiments/metrics/gini_coefficient_inv.py)| [Link](https://dl.acm.org/doi/abs/10.1145/3298689.3347040?casa_token=-QId0RoJsHgAAAAA:er_vhmem2f1h-_Yv4YJ3E0vXg6F-0tnu62c08l4g_9_TFmNDUEpBJTZQZgUniyH1fhEhkcWVUBWGPl8) | Diversity is represented as the Gini coefficient – a measure of distributional inequality. It is measured as the inverse of cumulative frequency that each item is recommended.
| [Users Coverage](irec/offline_experiments/metrics/users_coverage.py) | [Link](https://link.springer.com/article/10.1007/s13042-017-0762-9) | It represents the percentage of distinct users that are interested in at least k items recommended (k ≥ 1).
<!-- | [NDCG](reposit) | A diversity metric | [Link](link)  -->
<!-- | [F-Measure](reposit) | desc | [Link](link) -->
<!-- | [MAE](reposit) | desc | [Link](link)  -->
<!-- | [RMSE](reposit) | desc | [Link](link)  -->

## API

For writing anything new to the library (e.g., value function, agent, etc) read the [documentation](https://irec-org.github.io/irec/).

## Contributing 

All contributions are welcome! Just open a pull request.

## Related Projects

- [iRec-cmdline](https://github.com/irec-org/irec-cmdline)
