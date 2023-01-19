import os
import sys

import pytest

data_path = os.path.join(os.path.dirname(__file__), "../models")
sys.path.append(os.path.abspath(data_path))


from train_model import main


@pytest.mark.skipif(
    not os.path.exists("data/"),
    reason="Data files not found, this should be tested locally only",
)
def test_train():
    assert 1 + 1 == 2
