class Action:
    """Action."""

    pass


class UIAction(Action):
    """UIAction."""

    def __init__(self, user: int, item: int):
        """__init__.

        Args:
            user (int): user
            item (int): item
        """
        self.user = user
        self.item = item
