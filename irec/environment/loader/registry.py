from __future__ import annotations
from typing import List


class LoaderRegistry:

    from irec.environment.loader.train_test import TrainTestLoader
    from irec.environment.loader.full_data import DefaultLoader

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
