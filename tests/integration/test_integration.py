import os
import pytest

@pytest.mark.parametrize(
    "models, datasets, metrics",
    [
        (
            "MostPopular Random UCB",
            "MovieLens 100k",
            "Hits",
        ),
    ],
)
def test_integration(models, datasets, metrics):

    os.system(f"cd app && python3 run_agent_best.py --agents {models} --dataset_loaders '{datasets}' --evaluation_policy FixedInteraction --forced_run > out.txt")
    # print("\nOIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n")
    # os.system("ls")
    assert 2 == 3, "ok"
    # pass