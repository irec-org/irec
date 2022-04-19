from typing import List

class Action:
    """Action.
    
    An action represents the interaction between users and items
    """
    pass


class UserItemAction(Action):
    """UserItemAction.
    
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


class OneUserItemCollection(Action):
    """OneUserItemCollection.
    
        Used to represent the scenario where more than one item is recommended for a user

    """

    def __init__(self, user: int, items: List[int]) -> None:
        """__init__.

        Args:
            user (int): user
            items (List[int]): items

        Returns:
            None:
        """
        super().__init__()
        self.user = user
        self.items = items

    def __getitem__(self, key):
        if key == 0:
            return self.user
        elif key == 1:
            return self.items
        else:
            raise IndexError
