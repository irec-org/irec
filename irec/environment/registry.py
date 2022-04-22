from __future__ import annotations
from typing import List


class SplitRegistry:

    from irec.environment.split.randomised import Random
    from irec.environment.split.temporal import Temporal
    from irec.environment.split.global_timestamp import GlobalTimestampSplit

    _splitting = {
        "temporal": Temporal,
        "random": Random,
        "global": GlobalTimestampSplit,

    }

    @classmethod
    def all(cls: SplitRegistry) -> List[str]:
        return list(cls._splitting.keys())

    @classmethod
    def get(cls: SplitRegistry, name: str):
        return cls._splitting[name]


class FilterRegistry:

    from irec.environment.filter.filtering_by_items import FilteringByItems
    from irec.environment.filter.filtering_by_users import FilteringByUsers

    _filters = {
        "filter_users": FilteringByUsers,
        "filter_items": FilteringByItems,
    }

    @classmethod
    def all(cls: FilterRegistry) -> List[str]:
        return list(cls._filters.keys())

    @classmethod
    def get(cls: FilterRegistry, name: str):
        return cls._filters[name]


class LoaderRegistry:
    
    from irec.environment.loader.train_test import SplitData
    from irec.environment.loader.full_data import FullData

    # TODO: rename these parameters?
    _loader = {
        "TrainTestLoader": SplitData,
        "DefaultLoader": FullData,
    }

    @classmethod
    def all(cls: LoaderRegistry) -> List[str]:
        return list(cls._loader.keys())

    @classmethod
    def get(cls: LoaderRegistry, name: str):
        return cls._loader[name]
