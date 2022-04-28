from __future__ import annotations
from typing import List


class LoaderRegistry:

    from irec.environment.loader.train_test import SplitData
    from irec.environment.loader.full_data import FullData

    _loader = {
        "SplitData": SplitData,
        "FullData": FullData,
    }

    @classmethod
    def all(cls: LoaderRegistry) -> List[str]:
        return list(cls._loader.keys())

    @classmethod
    def get(cls: LoaderRegistry, name: str):
        return cls._loader[name]
