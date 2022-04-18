class Action:
    """Action."""

    pass


class UIAction(Action):
    """UIAction.
    
        Used to represent the scenario where one item is recommended for a user

    """

    def __init__(self, user: int, item: int):
        """__init__.

        Args:
            user (int): user
            item (int): item
        """
        self.user = user
        self.item = item
