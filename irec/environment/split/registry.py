from __future__ import annotations
from typing import List


class SplitRegistry:

    from irec.environment.split.randomised import Random
    from irec.environment.split.temporal import Temporal
    from irec.environment.split.global_timestamp import GlobalTimestampSplit
    from irec.environment.split.user_history import UserHistory

    _splitting = {
        "temporal": Temporal,
        "random": Random,
        "global": GlobalTimestampSplit,
        "user_history": UserHistory,
    }

    @classmethod
    def all(cls: SplitRegistry) -> List[str]:
        return list(cls._splitting.keys())

    @classmethod
    def get(cls: SplitRegistry, name: str):
        return cls._splitting[name]
