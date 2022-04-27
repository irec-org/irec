from __future__ import annotations
from typing import List

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
