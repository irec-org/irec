# Metric Evaluators

This module aims to guide the entire evalua- tion process over the logs from each iteration of the Evaluation Policy. As the iRec stores each execution log, the researcher can define how s/he would like to evaluate the actions selected by the recommendation model after all interactions.

The [Metric Evaluators](https://github.com/irec-org/irec/tree/update-info/irec/offline_experiments/metric_evaluators) supported by iRec are listed below.

| [Metric Evaluator](https://github.com/irec-org/irec/blob/update-info/irec/offline_experiments/metric_evaluators/base.py) | Description
| :---: | :--- |
| [Interaction](https://github.com/irec-org/irec/blob/update-info/irec/offline_experiments/metric_evaluators/interaction.py) | it evaluates the selected metrics’ overall interactions registered during the recommendation process. Given a scenario in which 100 interactions were performed, for instance, this strategy would evaluate each one separately.  
| [Stage Iterations](https://github.com/irec-org/irec/blob/update-info/irec/offline_experiments/metric_evaluators/stage_iterations.py) | it first aggregates some consecutive interactions in an interval to then evaluate the selected metrics over each group. For instance, in an execution of 10 interactions, the researcher can define two intervals to be evaluated by the system. 
| [Total](https://github.com/irec-org/irec/blob/update-info/irec/offline_experiments/metric_evaluators/total.py) | it evaluates the whole recommendation process as one unique procedure. For example, if certain items were recommended during 100 interactions, the metric will be calculated only at the 100th interaction.
| [User Cumulative Interaction](https://github.com/irec-org/irec/blob/update-info/irec/offline_experiments/metric_evaluators/user_cumulative_interaction.py) | it evaluates the interactions cumulatively from the first one until a specific value. In this way, the researcher can evaluate the accumulated result from the 1st to the 10th interaction, then from the 1st to the 15th interaction, and so on.

<!-- | [Iterations](https://github.com/irec-org/irec/blob/update-info/irec/offline_experiments/metric_evaluators/iterations.py) | short description  -->
<!-- | [Cumulative Interaction](https://github.com/irec-org/irec/blob/update-info/irec/offline_experiments/metric_evaluators/cumulative_interaction.py) | short description.
| [Cumulative](https://github.com/irec-org/irec/blob/update-info/irec/offline_experiments/metric_evaluators/cumulative.py) | short description   -->