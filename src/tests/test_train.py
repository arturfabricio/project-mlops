import os
import sys
import torch
import pytest

data_path = os.path.join(os.path.dirname(__file__), '../models')
sys.path.append(os.path.abspath(data_path))

from train_model import main

def test_train():
    assert 1+1 == 2
    

