class CandidateAction:
    """CandidateAction."""

    pass


class UICandidateAction(CandidateAction):
    """UICandidateAction."""

    def __init__(self, user: int, item: int):
        """__init__.

        Args:
            user (int): user
            item (int): item
        """
        self.user = user
        self.item = item
