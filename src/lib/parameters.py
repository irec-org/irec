from dataclasses import dataclass

@dataclass
class EvaluationPolicyParameters:
    interactions: int
    interaction_size: int

@dataclass   
class InteractorParameters:
    evaluation_policy: EvaluationPolicy

