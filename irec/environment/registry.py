from __future__ import annotations
from typing import List


class SplitRegistry:

    from irec.environment.split.randomised import Random
    from irec.environment.split.temporal import Temporal

    _splitting = {
        "temporal": Temporal,
        "random": Random,
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
    
    from irec.environment.load.train_test import TrainTestLoader
    from irec.environment.load.full_data import DefaultLoader

    _loader = {
        "TrainTestLoader": TrainTestLoader,
        "DefaultLoader": DefaultLoader,
    }

    @classmethod
    def all(cls: LoaderRegistry) -> List[str]:
        return list(cls._loader.keys())

    @classmethod
    def get(cls: LoaderRegistry, name: str):
        return cls._loader[name]
