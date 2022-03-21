from __future__ import annotations

from typing import List

from irec.environment.filter.filtering_by_items import FilteringByItems
from irec.environment.filter.filtering_by_users import FilteringByUsers
from irec.environment.split.randomised import Random
from irec.environment.split.temporal import Temporal


class SplitRegistry:

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
