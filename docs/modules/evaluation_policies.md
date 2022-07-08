# Evaluation Policies

This module is responsible for determining
how the recommendation agent will interact with the previously
defined environment. Basically, it implements the classical rein
forcement learning algorithm where the system:

(1) selects the target user;

(2) gets the action from the recommendation model;

(3) receives the feedback from the user to that specific action;

(4) updates the model’s knowledge with this reward

The [Evaluation Policies](https://github.com/irec-org/irec/tree/update-info/irec/offline_experiments/evaluation_policies) supported by iRec are listed below.

| [Evaluation Policy](https://github.com/irec-org/irec/blob/master/irec/evaluation_policies/EvaluationPolicy.py) | The base class for all evaluation policies.
| :---: | :--- |
| [FixedInteraction](https://github.com/irec-org/irec/blob/update-info/irec/offline_experiments/evaluation_policies/fixed_interaction.py) | Each user is randomly selected and each action will not be per- formed more than once for him/her. Each user will be selected for 𝑇 times. Thus, the system will perform 𝑇 × \|𝑈 \| iterations, where \|𝑈 \| is the number of distinct users available for the evaluation. The number 𝑇 is predefined by the researcher as a parameter.
| [Limited Interaction](https://github.com/irec-org/irec/blob/update-info/irec/offline_experiments/evaluation_policies/limited_interaction.py) | The system will perform new actions until it hits all items registered in the user historical. Theidea is to make an exhaustive experiment to observe which algo- rithm takes more time to reach all items previous rated by each user.