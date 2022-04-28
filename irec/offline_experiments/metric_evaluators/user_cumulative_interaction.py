from .cumulative_interaction import CumulativeInteraction

class UserCumulativeInteraction(CumulativeInteraction):
    @staticmethod
    def metric_summarize(users_metric_values):
        return users_metric_values