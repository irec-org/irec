from typing import List


class ActionCollection:
    """ActionCollection."""

    pass


class OneUserActionCollection(ActionCollection):
    """OneUserActionCollection."""

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
