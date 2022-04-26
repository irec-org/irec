from .CumulativeInteractionMetricEvaluator import CumulativeInteractionMetricEvaluator

class UserCumulativeInteractionMetricEvaluator(CumulativeInteractionMetricEvaluator):
    @staticmethod
    def metric_summarize(users_metric_values):
        return users_metric_values